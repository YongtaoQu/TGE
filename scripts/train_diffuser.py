import os
import sys

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import wandb

import diffuser.utils as utils
from diffuser.datasets.create_offline_dataset import create_mixed_dataset


def mixed_run_name(env_name: str, data_type: str, num_expert: int) -> str:
    return f"mixed_{env_name}_{data_type}_expert{num_expert}"


class Parser(utils.Parser):
    env_name: str = "walker2d"
    # Kept for diffuser.Parser savepath bootstrap; overwritten from env_name/data_type/N.
    dataset: str = "mixed_walker2d_medium_expert30"
    config: str = "diffuser.config.locomotion"
    horizon: int = 32
    data_type: str = "medium"
    num_expert_trajectories: int = 30
    save_freq: int = 100000


args = Parser().parse_args("diffusion")
utils.set_device(args.device)
wandb.init(project="TGE-diffuser", mode="disabled")

args.dataset = mixed_run_name(args.env_name, args.data_type, args.num_expert_trajectories)
args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name)
os.makedirs(args.savepath, exist_ok=True)

print(
    f"[train_diffuser] env={args.env_name} data_type={args.data_type} "
    f"N={args.num_expert_trajectories} H={args.horizon} device={args.device}"
)
print(f"[train_diffuser] run={args.dataset}")
print(f"[train_diffuser] savepath={args.savepath}")

dataset_dict = create_mixed_dataset(
    args.env_name, args.data_type, args.num_expert_trajectories
)

dataset = utils.Config(
    args.loader,
    savepath=(args.savepath, "dataset_config.pkl"),
    env=args.dataset,
    dataset_dict=dataset_dict,
    horizon=args.horizon,
    max_path_length=args.max_path_length,
    normalizer=args.normalizer,
    use_padding=args.use_padding,
)()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

model = utils.Config(
    args.model,
    savepath=(args.savepath, "model_config.pkl"),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
)()

diffusion = utils.Config(
    args.diffusion,
    savepath=(args.savepath, "diffusion_config.pkl"),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)(model)

trainer = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, "trainer_config.pkl"),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=0,
    save_freq=args.save_freq,
    label_freq=args.save_freq,
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    eval_freq=0,
    device=args.device,
)(diffusion, dataset, renderer=None)

n_train_steps = int(args.n_train_steps)
n_steps_per_epoch = int(args.n_steps_per_epoch)
n_epochs = n_train_steps // n_steps_per_epoch
remainder = n_train_steps % n_steps_per_epoch

for i in range(n_epochs):
    print(f"Epoch {i} / {n_epochs} | {args.savepath}")
    trainer.train(n_train_steps=n_steps_per_epoch)
if remainder:
    trainer.train(n_train_steps=remainder)

print(f"[train_diffuser] Saving final checkpoint at step {n_train_steps} (trainer.step={trainer.step})")
trainer.save(n_train_steps)
