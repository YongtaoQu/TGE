import os
import sys
import math
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple, Union, Optional
from pathlib import Path

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import chex
import d4rl
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import wandb
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from tqdm.auto import trange

from make_envs import make_env

default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


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
    env: str = "hopper-random-v2"
    seed: int = 0
    eval_freq: int = int(2e4)
    n_episodes: int = 100
    max_timesteps: int = int(300000)
    checkpoints_path: Optional[str] = None
    load_model: str = ""

    actor_learning_rate: float = 5e-4
    critic_learning_rate: float = 5e-4
    hidden_dim: int = 256
    actor_n_hiddens: int = 3
    critic_n_hiddens: int = 3
    gamma: float = 0.99
    tau: float = 5e-3
    actor_bc_coef: float = 0.4
    critic_bc_coef: float = 1.0
    actor_ln: bool = True
    critic_ln: bool = True
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    normalize_q: bool = True
    batch_size: int = 4096
    weight_decay: float = 1e-3
    actor_dropout: float = 0.0

    normalize_reward: bool = False
    normalize: bool = False
    buffer_size: int = 2_000_000

    verbose: bool = True
    project: str = "TGE"
    group: str = "rebrac"
    name: str = None

    expert_trajectories: int = 30
    horizon: Union[Optional[int], str] = 32
    labeled_dataset_path: Optional[str] = None
    max_episode_steps: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.horizon, str) and self.horizon.lower() == "none":
            self.horizon = None

        horizon_suffix = f"_H{self.horizon}" if self.horizon is not None else ""
        setting_tag = f"{self.env}_N{self.expert_trajectories}{horizon_suffix}_seed{self.seed}"

        if self.name is None or self.name == "None":
            self.name = f"{setting_tag}-{str(uuid.uuid4())[:8]}"
        if self.group == "rebrac":
            self.group = f"{self.env}_N{self.expert_trajectories}{horizon_suffix}"

        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

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

        if any(s in self.env.lower() for s in ("door", "pen", "hammer", "relocate")):
            self.actor_dropout = 0.1
            self.actor_ln = False
            if "human" in self.env.lower():
                self.actor_bc_coef = 1.0

        if self.env == "ant-medium-v2":
            self.actor_bc_coef = 0.1

        self.max_episode_steps = get_max_episode_steps(self.env)


def pytorch_init(fan_in: float) -> Callable:
    bound = math.sqrt(1 / fan_in)
    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )
    return _init


def uniform_init(bound: float) -> Callable:
    def _init(key: jax.random.PRNGKey, shape: Tuple, dtype: type) -> jax.Array:
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )
    return _init


class DetActor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, state: jax.Array, training: bool = False) -> jax.Array:
        s_d, h_d = state.shape[-1], self.hidden_dim
        x = state
        
        for i in range(self.n_hiddens):
            in_dim = s_d if i == 0 else h_d
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(in_dim),
                bias_init=nn.initializers.constant(0.1),
            )(x)
            x = nn.relu(x)
            if self.layernorm:
                x = nn.LayerNorm()(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = nn.Dense(
            self.action_dim,
            kernel_init=uniform_init(1e-3),
            bias_init=uniform_init(1e-3),
        )(x)
        return nn.tanh(x)


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
        x = jnp.hstack([state, action])
        
        for i in range(self.n_hiddens):
            in_dim = (s_d + a_d) if i == 0 else h_d
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(in_dim),
                bias_init=nn.initializers.constant(0.1),
            )(x)
            x = nn.relu(x)
            if self.layernorm:
                x = nn.LayerNorm()(x)
        
        x = nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))(x)
        return x.squeeze(-1)


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 10
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics,
        )
        return ensemble(self.hidden_dim, self.layernorm, self.n_hiddens)(state, action)


def compute_mean_std(states: jax.Array, eps: float) -> Tuple[jax.Array, jax.Array]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
    return (states - mean) / std
    

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
    lengths.append(ep_len)
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d", "ant", "door", "hammer", "pen", "relocate")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


@chex.dataclass
class ReplayBuffer:
    data: Dict[str, jax.Array] = None
    mean: float = 0
    std: float = 1

    def load_dataset(self, dataset: Dict[str, np.ndarray], normalize: bool = False):
        observations = jnp.asarray(dataset["observations"], dtype=jnp.float32)
        actions = jnp.asarray(dataset["actions"], dtype=jnp.float32)
        next_observations = jnp.asarray(dataset["next_observations"], dtype=jnp.float32)
        rewards = jnp.asarray(dataset["rewards"], dtype=jnp.float32)
        terminals = jnp.asarray(dataset["terminals"], dtype=jnp.float32)
        
        next_actions = np.roll(dataset["actions"], -1, axis=0)
        is_final_step = np.logical_or(dataset["terminals"], dataset.get("timeouts", np.zeros_like(terminals)))
        next_actions[is_final_step == 1] = dataset["actions"][is_final_step == 1]

        buffer = {
            "states": observations,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_observations,
            "next_actions": jnp.asarray(next_actions, dtype=jnp.float32),
            "dones": terminals,
        }

        if normalize:
            self.mean, self.std = compute_mean_std(buffer["states"], eps=1e-3)
            buffer["states"] = normalize_states(buffer["states"], self.mean, self.std)
            buffer["next_states"] = normalize_states(buffer["next_states"], self.mean, self.std)
            
        self.data = buffer

    @property
    def size(self) -> int:
        return self.data["states"].shape[0]

    def sample_batch(self, key: jax.random.PRNGKey, batch_size: int) -> Dict[str, jax.Array]:
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.size)
        return jax.tree_map(lambda arr: arr[indices], self.data)


@chex.dataclass(frozen=True)
class Metrics:
    accumulators: Dict[str, Tuple[jax.Array, jax.Array]]

    @staticmethod
    def create(metrics: Sequence[str]) -> "Metrics":
        init_metrics = {key: (jnp.array([0.0]), jnp.array([0.0])) for key in metrics}
        return Metrics(accumulators=init_metrics)

    def update(self, updates: Dict[str, jax.Array]) -> "Metrics":
        new_accumulators = deepcopy(self.accumulators)
        for key, value in updates.items():
            acc, steps = new_accumulators[key]
            new_accumulators[key] = (acc + value, steps + 1)
        return self.replace(accumulators=new_accumulators)

    def compute(self) -> Dict[str, np.ndarray]:
        return {k: np.array(v[0] / v[1]) for k, v in self.accumulators.items()}


def wrap_env(env: gym.Env, state_mean: Union[np.ndarray, float] = 0.0, 
             state_std: Union[np.ndarray, float] = 1.0, reward_scale: float = 1.0) -> gym.Env:
    env = gym.wrappers.TransformObservation(env, lambda s: (s - state_mean) / state_std)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, lambda r: reward_scale * r)
    return env


def evaluate(env: gym.Env, params: jax.Array, action_fn: Callable, 
             num_episodes: int, seed: int, verbose: bool = True) -> np.ndarray:
    n_episodes_per_seed = 10
    n_seeds = num_episodes // n_episodes_per_seed
    rng = np.random.RandomState(seed)
    eval_seeds = rng.randint(0, 10_000_000, size=n_seeds)
    all_episode_rewards = []
    
    for seed_idx, eval_seed in enumerate(eval_seeds):
        env.seed(int(eval_seed))
        seed_rewards = []
        for _ in range(n_episodes_per_seed):
            obs, done = env.reset(), False
            total_reward = 0.0
            while not done:
                action = np.asarray(jax.device_get(action_fn(params, obs)))
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            seed_rewards.append(total_reward)
        all_episode_rewards.extend(seed_rewards)
    return np.array(all_episode_rewards)


class CriticTrainState(TrainState):
    target_params: FrozenDict


class ActorTrainState(TrainState):
    target_params: FrozenDict


def update_actor(key: jax.random.PRNGKey, actor: TrainState, critic: TrainState, 
                 batch: Dict[str, jax.Array], beta: float, tau: float, 
                 normalize_q: bool, metrics: Metrics) -> Tuple[jax.random.PRNGKey, TrainState, TrainState, Metrics]:
    key, actor_dropout_key = jax.random.split(key, 2)

    def actor_loss_fn(params: jax.Array) -> Tuple[jax.Array, Metrics]:
        actions = actor.apply_fn(params, batch["states"], training=True, rngs={'dropout': actor_dropout_key})
        bc_mse = ((actions - batch["actions"]) ** 2).sum(-1)
        weights = jax.lax.stop_gradient(batch["rewards"].flatten())
        weighted_bc_penalty = bc_mse * weights

        q_values = critic.apply_fn(critic.params, batch["states"], actions).min(0)
        
        lmbda = 1.0
        if normalize_q:
            q_mean = jax.numpy.abs(q_values).mean()
            lmbda = jnp.clip(jax.lax.stop_gradient(1 / (q_mean + 1e-4)), 1e-5, 1.0)
            
        loss = (beta * weighted_bc_penalty - lmbda * q_values).mean()
        new_metrics = metrics.update({
            "actor_loss": loss,
            "bc_mse_policy": bc_mse.mean(),
            "weighted_bc_penalty": weighted_bc_penalty.mean(),
            "action_mse": ((actions - batch["actions"]) ** 2).mean(),
        })
        return loss, new_metrics

    grads, new_metrics = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)
    new_actor = new_actor.replace(target_params=optax.incremental_update(actor.params, actor.target_params, tau))
    new_critic = critic.replace(target_params=optax.incremental_update(critic.params, critic.target_params, tau))
    return key, new_actor, new_critic, new_metrics


def update_critic(key: jax.random.PRNGKey, actor: TrainState, critic: CriticTrainState, 
                  batch: Dict[str, jax.Array], gamma: float, beta: float, 
                  policy_noise: float, noise_clip: float, metrics: Metrics) -> Tuple[jax.random.PRNGKey, TrainState, Metrics]:
    key, actions_key = jax.random.split(key, 2)
    next_actions = actor.apply_fn(actor.target_params, batch["next_states"], training=False)
    noise = jnp.clip(jax.random.normal(actions_key, next_actions.shape) * policy_noise, -noise_clip, noise_clip)
    next_actions = jnp.clip(next_actions + noise, -1, 1)
    
    bc_penalty = ((next_actions - batch["next_actions"]) ** 2).mean(-1)
    next_q = critic.apply_fn(critic.target_params, batch["next_states"], next_actions).min(0)
    target_q = batch["rewards"].flatten() + (1 - batch["dones"].flatten()) * gamma * (next_q - beta * bc_penalty)

    def critic_loss_fn(critic_params: jax.Array) -> Tuple[jax.Array, jax.Array]:
        q = critic.apply_fn(critic_params, batch["states"], batch["actions"])
        loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
        return loss, q.min(0).mean()

    (loss, q_min), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)
    return key, new_critic, metrics.update({"critic_loss": loss, "q_min": q_min})


def update_td3(key: jax.random.PRNGKey, actor: TrainState, critic: CriticTrainState, 
               batch: Dict[str, Any], metrics: Metrics, gamma: float, 
               actor_bc_coef: float, critic_bc_coef: float, tau: float, 
               policy_noise: float, noise_clip: float, normalize_q: bool) -> Tuple[jax.random.PRNGKey, TrainState, TrainState, Metrics]:
    key, new_critic, new_metrics = update_critic(key, actor, critic, batch, gamma, critic_bc_coef, policy_noise, noise_clip, metrics)
    key, new_actor, new_critic, new_metrics = update_actor(key, actor, new_critic, batch, actor_bc_coef, tau, normalize_q, new_metrics)
    return key, new_actor, new_critic, new_metrics


@pyrallis.wrap()
def main(config: TrainConfig):
    if "cuda:" in config.device and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.device.split(":")[-1]

    wandb.init(config=asdict(config), project=config.project, group=config.group, name=config.name, id=str(uuid.uuid4()))
    
    data = np.load(config.labeled_dataset_path, allow_pickle=True)
    dataset = {k: data[k] for k in ['observations', 'actions', 'rewards', 'next_observations', 'terminals']}

    if config.normalize_reward:
        modify_reward(dataset, config.env, max_episode_steps=config.max_episode_steps)

    buffer = ReplayBuffer()
    buffer.load_dataset(dataset, normalize=config.normalize)

    key = jax.random.PRNGKey(seed=config.seed)
    key, actor_key, critic_key = jax.random.split(key, 3)

    eval_env = wrap_env(make_env(config.env, monitor=False), buffer.mean, buffer.std)
    eval_env.seed(config.seed)
    
    init_state, init_action = buffer.data["states"][0][None, ...], buffer.data["actions"][0][None, ...]

    actor_module = DetActor(action_dim=init_action.shape[-1], hidden_dim=config.hidden_dim, 
                            layernorm=config.actor_ln, n_hiddens=config.actor_n_hiddens, dropout_rate=config.actor_dropout)
    actor = ActorTrainState.create(apply_fn=actor_module.apply, 
                                   params=actor_module.init(actor_key, init_state, training=False),
                                   target_params=actor_module.init(actor_key, init_state, training=False),
                                   tx=optax.adamw(optax.cosine_decay_schedule(config.actor_learning_rate, config.max_timesteps), weight_decay=config.weight_decay))

    critic_module = EnsembleCritic(hidden_dim=config.hidden_dim, num_critics=2, layernorm=config.critic_ln, n_hiddens=config.critic_n_hiddens)
    critic = CriticTrainState.create(apply_fn=critic_module.apply, 
                                     params=critic_module.init(critic_key, init_state, init_action),
                                     target_params=critic_module.init(critic_key, init_state, init_action),
                                     tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(optax.cosine_decay_schedule(config.critic_learning_rate, config.max_timesteps), weight_decay=config.weight_decay)))

    metrics_to_log = ["critic_loss", "q_min", "actor_loss", "bc_mse_policy", "weighted_bc_penalty", "action_mse"]
    carry = {"key": key, "actor": actor, "critic": critic, "buffer": buffer, "metrics": Metrics.create(metrics_to_log)}

    @jax.jit
    def train_step(i, carry):
        key, batch_key = jax.random.split(carry["key"])
        batch = carry["buffer"].sample_batch(batch_key, batch_size=config.batch_size)
        
        def full_update():
            return update_td3(key, carry["actor"], carry["critic"], batch, carry["metrics"], 
                              config.gamma, config.actor_bc_coef, config.critic_bc_coef, 
                              config.tau, config.policy_noise, config.noise_clip, config.normalize_q)

        def fast_update():
            _, nc, nm = update_critic(key, carry["actor"], carry["critic"], batch, config.gamma, config.critic_bc_coef, config.policy_noise, config.noise_clip, carry["metrics"])
            return key, carry["actor"], nc, nm

        key, new_actor, new_critic, new_metrics = jax.lax.cond(i % config.policy_freq == 0, full_update, fast_update)
        return {**carry, "key": key, "actor": new_actor, "critic": new_critic, "metrics": new_metrics}

    @jax.jit
    def actor_action_fn(params: jax.Array, obs: jax.Array):
        return actor.apply_fn(params, obs, training=False)

    for t in range(int(config.max_timesteps)):
        carry = train_step(t, carry)
        
        if (t + 1) % 1000 == 0:
            wandb.log({f"ReBRAC/{k}": float(v) for k, v in carry["metrics"].compute().items()}, step=t+1)
            carry["metrics"] = Metrics.create(metrics_to_log)

        if t % config.eval_freq == 0:
            eval_scores = evaluate(
                eval_env,
                carry["actor"].params,
                actor_action_fn,
                config.n_episodes,
                config.seed,
            )
            wandb.log({"d4rl_normalized_score": eval_env.get_normalized_score(eval_scores.mean()) * 100.0}, step=t+1)

if __name__ == "__main__":
    main()