import copy
import os
import random
import uuid
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)
from make_envs import make_env

TensorBatch = List[torch.Tensor]

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

def get_max_episode_steps(env_name: str) -> int:
    if "pen" in env_name:
        return 100
    elif "door" in env_name or "hammer" in env_name or "relocate" in env_name:
        return 200
    elif any(s in env_name for s in ("halfcheetah", "hopper", "walker2d", "ant")):
        return 1000
    elif "kitchen" in env_name:
        return 280
    else:
        raise ValueError(f"Unknown environment: {env_name}")

@dataclass
class TrainConfig:
    device: str = "cuda:0"
    env: str = "walker2d-medium-v0"
    seed: int = 0
    eval_freq: int = int(5e3)
    n_episodes: int = 100
    max_timesteps: int = int(5e5)
    checkpoints_path: Optional[str] = None
    load_model: str = ""

    buffer_size: int = 2_000_000
    batch_size: int = 1024
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 0.5
    iql_tau: float = 0.8
    iql_deterministic: bool = False
    normalize: bool = True
    normalize_reward: bool = False
    max_episode_steps: Optional[int] = None
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    weight_decay: float = 1e-4
    actor_dropout: Optional[float] = 0.0
    min_lr: float = 1e-6

    verbose: bool = True
    project: str = "TGE"
    group: str = "iql"
    name: str = None

    expert_trajectories: int = 30
    horizon: Union[Optional[int], str] = 32
    labeled_dataset_path: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
        
        if isinstance(self.horizon, str) and self.horizon.lower() == "none":
            self.horizon = None

        if self.labeled_dataset_path is None:
            horizon_val = self.horizon if self.horizon is not None else 32
            exp_folder_name = f"{self.env}_N{self.expert_trajectories}"
            self.labeled_dataset_path = os.path.join(
                _ROOT_DIR,
                "labeled_dataset",
                f"H{horizon_val}",
                exp_folder_name,
                "labeled_dataset.npz",
            )
        
        horizon_suffix = f"-H{self.horizon}" if self.horizon is not None else ""
        if self.group == "iql":
            self.group = f"{self.env}-{self.expert_trajectories}experts{horizon_suffix}"
            
        if any(s in self.env.lower() for s in ("door", "pen", "hammer", "relocate")):
            self.actor_dropout = 0.1

        if self.max_episode_steps is None:
            self.max_episode_steps = get_max_episode_steps(self.env)

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std

def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cuda",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        raise NotImplementedError

def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )

@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int, verbose: bool = True
) -> np.ndarray:
    actor.eval()
    all_episode_rewards = []
    
    n_episodes_per_seed = 10
    n_seeds = n_episodes // n_episodes_per_seed
    seeds = np.random.randint(0, 10_000_000, size=n_seeds)
    for seed_idx, seed in enumerate(seeds):
        env.seed(int(seed))
        seed_rewards = []
        
        for _ in range(n_episodes_per_seed):
            state, done = env.reset(), False
            episode_reward = 0.0
            while not done:
                action = actor.act(state, device)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
            seed_rewards.append(episode_reward)
        
        all_episode_rewards.extend(seed_rewards)
        mean, std = np.mean(seed_rewards), np.std(seed_rewards)
        norm_mean = env.get_normalized_score(mean) * 100.0
        if verbose:
            print(f"Seed {seed_idx}: {mean:.1f}±{std:.1f}, Normalized: {norm_mean:.1f}")

    actor.train()
    return np.asarray(all_episode_rewards)

def get_max_episode_steps(env_name: str) -> int:
    """Return max_episode_steps based on environment name."""
    if "pen" in env_name:
        return 100
    elif "door" in env_name or "hammer" in env_name or "relocate" in env_name:
        return 200
    elif any(s in env_name for s in ("halfcheetah", "hopper", "walker2d", "ant")):
        return 1000
    elif "kitchen" in env_name:
        return 280
    else:
        raise ValueError(f"Unknown environment: {env_name}")

def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)

def modify_reward(dataset, env_name, max_episode_steps=None):
    if max_episode_steps is None:
        max_episode_steps = get_max_episode_steps(env_name)
    
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d", "door", "pen", "hammer", "relocate")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)

class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
        use_layernorm: bool = False,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_layernorm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()

class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )

class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, n_qs: int = 2, hidden_dim: int = 256, n_hidden: int = 2,
        use_layernorm: bool = True,  # Use LayerNorm to prevent Q-value explosion
    ):
        super().__init__()
        self.n_qs = n_qs
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.qs = nn.ModuleList([MLP(dims, squeeze_output=True, use_layernorm=use_layernorm) for _ in range(n_qs)])

    def all(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        sa = torch.cat([state, action], 1)
        return tuple(q(sa) for q in self.qs)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        qs = self.all(state, action)
        qs_tensor = torch.stack(qs, dim=0)  # [n_qs, batch, ...]
        return torch.min(qs_tensor, dim=0)[0]

class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2, use_layernorm: bool = True):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True, use_layernorm=use_layernorm)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)

class ImplicitQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 1.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
        min_lr: float = 1e-6,
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, int(max_steps))
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        
        v_grad_norm = torch.nn.utils.clip_grad_norm_(self.vf.parameters(), max_norm=10.0)
        
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.all(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        
        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.qf.parameters(), max_norm=10.0)
        
        self.q_optimizer.step()

        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        
        log_dict["exp_adv_max"] = exp_adv.max().item()
        
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        
        
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])
        
        self.total_it = state_dict["total_it"]

@pyrallis.wrap()
def train(config: TrainConfig):
    env = make_env(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

   
    data = np.load(config.labeled_dataset_path, allow_pickle=True)
    observations = data['observations']
    actions = data['actions']
    next_observations = data['next_observations']
    rewards = data['rewards']
    terminals = data['dones']
    
    if config.verbose:
        print(f"[IQL] Loaded dataset from: {config.labeled_dataset_path}")

    dataset = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'next_observations': next_observations,
        'terminals': terminals,
    }

    if config.normalize_reward:
        modify_reward(dataset, config.env, max_episode_steps=config.max_episode_steps)
    
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None and config.checkpoints_path.strip() != "":
        if config.verbose:
            print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    seed = config.seed
    set_seed(seed, env)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    v_optimizer = torch.optim.AdamW(v_network.parameters(), lr=config.vf_lr, weight_decay=config.weight_decay)
    q_optimizer = torch.optim.AdamW(q_network.parameters(), lr=config.qf_lr, weight_decay=config.weight_decay)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=config.actor_lr, weight_decay=config.weight_decay)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
        "min_lr": config.min_lr,
    }

    if config.verbose:
        print("---------------------------------------")
        print(f"Training IQL, Env: {config.env}, Seed: {seed}")
        print("---------------------------------------")

    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    best_score = -float('inf')
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        trainer.total_it += 1  # Increment after training
        wandb.log(log_dict, step=trainer.total_it)
        if (t + 1) % max(1000, config.eval_freq // 10) == 0 and config.verbose:
            v_loss = log_dict.get("value_loss", float('nan'))
            q_loss = log_dict.get("q_loss", float('nan'))
            actor_loss = log_dict.get("actor_loss", float('nan'))
            print(f"[IQL][step {trainer.total_it}] v_loss={v_loss:.6f} q_loss={q_loss:.6f} actor_loss={actor_loss:.6f}")

        if (t) % config.eval_freq == 0:
            if config.verbose:
                print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
                verbose=config.verbose,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            
            is_best = normalized_eval_score > best_score
            if is_best:
                best_score = normalized_eval_score
            
            if config.verbose:
                print("---------------------------------------")
                print(f"Evaluation (10 seeds × 10 episodes = 100 total episodes):")
                print(f"Return: Mean={eval_score:.3f}, Std={eval_scores.std():.3f}, Max={eval_scores.max():.3f}, Min={eval_scores.min():.3f}")
                d4rl_scores = env.get_normalized_score(eval_scores) * 100.0
                print(f"D4RL:   Mean=\033[94m{normalized_eval_score:.3f}\033[0m, Std=\033[94m{d4rl_scores.std():.3f}\033[0m, Max={env.get_normalized_score(eval_scores.max()) * 100.0:.3f}, Min={env.get_normalized_score(eval_scores.min()) * 100.0:.3f}")
                low_d4rl_scores = eval_scores[d4rl_scores < 30]
                if low_d4rl_scores.size > 0:
                    print(f"Raw episode scores with D4RL score < 30 (count={low_d4rl_scores.size})")
                print("---------------------------------------")
            wandb.log(
                {
                    "d4rl_normalized_score": normalized_eval_score,
                    "d4rl_normalized_score_std": (env.get_normalized_score(eval_scores) * 100.0).std(),
                },
                step=trainer.total_it
            )

if __name__ == "__main__":
    train()