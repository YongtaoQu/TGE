import os
import sys

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import argparse
import glob
import pickle

import numpy as np
import torch
import gym
import d4rl

from diffuser.datasets.create_offline_dataset import create_mixed_dataset
from diffuser.utils.serialization import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Label datasets with TGE reward scores.")
    parser.add_argument("--env_name", type=str, default="walker2d")
    parser.add_argument(
        "--data_type",
        type=str,
        default="medium",
        choices=["random", "medium", "random-medium", "human", "cloned", "partial", "mixed"],
    )
    parser.add_argument("--num_expert", type=int, default=30)
    parser.add_argument("--score_temperature", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output_path", type=str, default="labeled_dataset.npz")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--embed_batch_size", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument(
        "--logbase",
        type=str,
        default=None,
        help="Diffuser log root (default: <repo>/logs).",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory with state_*.pt + model/diffusion configs.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Explicit state_*.pt (uses parent dir as checkpoint_dir if needed).",
    )
    parser.add_argument(
        "--expert_data_path",
        type=str,
        default=None,
        help="Expert reference pickle for TGE scoring (default: expert_data/{env}_expert.pkl).",
    )
    parser.add_argument(
        "--mismatch",
        action="store_true",
        help="Tag labeled output as mismatched dynamics (folder: {env}_mismatched_N{N}).",
    )
    return parser.parse_args()


def _latest_numeric_state_pt(checkpoint_parent):
    scored = []
    for ckpt in glob.glob(os.path.join(checkpoint_parent, "state_*.pt")):
        stem = os.path.basename(ckpt).split("_", 1)[1].split(".")[0]
        if stem.isdigit():
            scored.append((int(stem), ckpt))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return scored[-1][1]


def resolve_diffuser_checkpoint(env_name, data_type, num_expert, horizon=32, logbase=None):
    """Resolve logs/mixed_{env}_{type}_expert{N}/diffusion/defaults_H{H}_T20/state_*.pt."""
    logbase = logbase or os.path.join(_ROOT_DIR, "logs")
    env_key = env_name.split("-")[0]
    run_dir = f"mixed_{env_key}_{data_type}_expert{num_expert}"
    path = os.path.join(logbase, run_dir, "diffusion", f"defaults_H{horizon}_T20")
    latest = _latest_numeric_state_pt(path)
    if latest is None:
        raise FileNotFoundError(
            f"No state_*.pt under {path}. Train diffuser first or pass --checkpoint_dir/--checkpoint_path."
        )
    return path, latest


def load_diffuser_checkpoint(diffuser, checkpoint_path, device="cuda:0"):
    print(f"Loading diffuser checkpoint from {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location=device)
    diffuser.load_state_dict(data["model"])
    return diffuser

def split_offline_and_expert(dataset):
    mask = dataset["is_expert"].astype(bool)

    def subset(mask_value):
        return {
            key: value[mask_value]
            for key, value in dataset.items()
            if isinstance(value, np.ndarray) and value.shape[0] == mask_value.shape[0]
        }

    offline = subset(~mask)
    expert = subset(mask)
    return offline, expert

def make_offline_env_name(env_name, data_type):
    suffix_map = {
        "random": "-random-v2",
        "medium": "-medium-v2",
        "random-medium": "-random-medium-v2",
        "human": "-human-v0",
        "cloned": "-cloned-v0",
        "partial": "-partial-v0",
        "mixed": "-mixed-v0",
    }
    if data_type not in suffix_map:
        raise ValueError(f"Unsupported data_type '{data_type}' for env id construction.")
    return f"{env_name}{suffix_map[data_type]}"

def load_expert_reference(env_name, expert_data_path=None):
    base_env_name = env_name.split('-')[0]

    if expert_data_path is None:
        expert_data_dir = os.path.join(_ROOT_DIR, "expert_data")
        expert_data_path = os.path.join(expert_data_dir, f"{base_env_name}_expert.pkl")

    print(f"Loading expert reference from {expert_data_path}")
    if not os.path.exists(expert_data_path):
        raise FileNotFoundError(f"Expert data file not found at {expert_data_path}.")

    with open(expert_data_path, 'rb') as f:
        expert_traj = pickle.load(f)

    if 'observations' in expert_traj:
        obs = expert_traj['observations']
    elif 'states' in expert_traj:
        obs = expert_traj['states']
    else:
        raise KeyError(f"Expert data must have 'observations' or 'states' key.")

    if isinstance(obs, list):
        obs = np.concatenate([np.asarray(x) for x in obs], axis=0)
    else:
        obs = np.asarray(obs)
    if obs.ndim == 3:
        obs = obs[0]
    if obs.ndim != 2:
        raise ValueError(f"Expert observations must be 2D after unwrap, got shape {obs.shape}")

    return {"observation": obs.astype(np.float32)}

def compute_embeddings(
    diffuser,
    observations,
    device="cuda",
    dones=None,
    desc="Embedding",
    batch_size=512,
):
    """Encode per-timestep state embeddings with batched diffuser forward passes.

    Numerically equivalent to the previous per-step (batch_size=1) loop.
    """
    model = diffuser.model
    window_len = diffuser.horizon
    state_dim = diffuser.observation_dim
    action_dim = diffuser.action_dim
    batch_size = max(1, int(batch_size))

    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    n_steps = obs.shape[0]

    episode_ends = np.zeros(n_steps, dtype=int)
    if dones is not None:
        dones_bool = np.asarray(dones, dtype=bool)
        done_idxs = np.where(dones_bool)[0]
        if len(done_idxs) == 0 or done_idxs[-1] != n_steps - 1:
            done_idxs = np.append(done_idxs, n_steps - 1)
        start_ptr = 0
        for end_ptr in done_idxs:
            episode_ends[start_ptr : end_ptr + 1] = end_ptr + 1
            start_ptr = end_ptr + 1
    else:
        episode_ends[:] = n_steps

    windows = np.zeros((n_steps, window_len, action_dim + state_dim), dtype=np.float32)
    for t in range(n_steps):
        if t % 100000 == 0:
            print(f"{desc} build windows {t} / {n_steps}")
            sys.stdout.flush()
        max_idx = episode_ends[t]
        start = t
        end = min(start + window_len, max_idx)
        seg_obs = obs[start:end]
        pad_len = window_len - seg_obs.shape[0]
        if pad_len > 0:
            pad_obs = np.repeat(seg_obs[-1:], pad_len, axis=0)
            win_obs = np.concatenate([seg_obs, pad_obs], axis=0)
        else:
            win_obs = seg_obs
        windows[t, :, action_dim:] = win_obs

    model.eval()
    print(f"{desc} encode {n_steps} steps in batches of {batch_size}")
    sys.stdout.flush()
    all_z_chunks = []
    with torch.no_grad():
        for start in range(0, n_steps, batch_size):
            end = min(start + batch_size, n_steps)
            if start == 0 or (start % (batch_size * 20) == 0):
                print(f"{desc} {start} / {n_steps} (batch={batch_size})")
                sys.stdout.flush()
            batch = torch.from_numpy(windows[start:end]).to(device, non_blocking=True)
            z = model.encode_state_embedding(batch, state_dim)
            all_z_chunks.append(z.detach().cpu().numpy())
    return np.concatenate(all_z_chunks, axis=0)

def score_offline_against_reference(
    mixed_dataset,
    reference,
    diffuser,
    temperature=1.0,
    device="cuda",
    k=10,
    task_description="",
    embed_batch_size=512,
):
    print(f"Scoring with TGE Trajectory Embeddings (k={k}, temp={temperature})...")
    
    dones = mixed_dataset.get("dones")
    
    desc_off = f"Embed Offline [{task_description}]" if task_description else "Embed Offline"
    offline_embeddings = compute_embeddings(
        diffuser,
        mixed_dataset["observations"],
        device=device,
        dones=dones,
        desc=desc_off,
        batch_size=embed_batch_size,
    )

    desc_ref = f"Embed Reference [{task_description}]" if task_description else "Embed Reference"
    reference_embedding = compute_embeddings(
        diffuser,
        reference["observation"],
        device=device,
        desc=desc_ref,
        batch_size=embed_batch_size,
    )

    off_emb_torch = torch.nn.functional.normalize(torch.from_numpy(offline_embeddings).to(device), p=2, dim=1)
    ref_emb_torch = torch.nn.functional.normalize(torch.from_numpy(reference_embedding).to(device), p=2, dim=1)

    N = off_emb_torch.shape[0]
    rewards = torch.zeros(N, device=device)
    batch_size = 4096 

    for i in range(0, N, batch_size):
        if i == 0 or (i % 10000 < batch_size):
            print(f"Computing Scores: {i} / {N}")
            sys.stdout.flush()

        end = min(i + batch_size, N)
        batch_off = off_emb_torch[i:end]
        dists = torch.cdist(batch_off, ref_emb_torch, p=2) 
        topk_dists, _ = torch.topk(dists, k=k, largest=False, dim=1)
        
        log_dists = -torch.log(1.0 + topk_dists / temperature)
        rewards[i:end] = torch.mean(log_dists, dim=1)

    rewards_np = rewards.cpu().numpy()
    r_min, r_max = rewards_np.min(), rewards_np.max()
    print(f"Raw Stats: Min={r_min:.4f}, Max={r_max:.4f}, Mean={rewards_np.mean():.4f}")
    
    rewards_scaled = np.clip((rewards_np - r_min) / (r_max - r_min + 1e-8), 0.0, 1.0)

    labeled_dataset = dict(mixed_dataset)
    labeled_dataset["rewards"] = rewards_scaled.astype(np.float32)
    return labeled_dataset

def main():
    args = parse_args()
    env_id = make_offline_env_name(args.env_name, args.data_type)
    
    mismatch_tag = "_mismatched" if args.mismatch else ""
    exp_folder_name = f"{env_id}{mismatch_tag}_N{args.num_expert}"
    if args.score_temperature != 1.0 or args.k != 10:
        exp_folder_name += f"_T{args.score_temperature}_k{args.k}"

    output_dir = os.path.join(_ROOT_DIR, "labeled_dataset", f"H{args.horizon}", exp_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.output_path == "labeled_dataset.npz":
        args.output_path = os.path.join(output_dir, "labeled_dataset.npz")

    log_path = args.output_path.replace(".npz", ".log")
    
    class Tee(object):
        def __init__(self, name, mode):
            self.file = open(name, mode)
            self.stdout = sys.stdout
        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)
        def flush(self):
            self.file.flush()
            self.stdout.flush()
        def __del__(self):
            self.file.close()

    sys.stdout = Tee(log_path, 'w')
    sys.stderr = sys.stdout 

    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        checkpoint_dir = args.checkpoint_dir or os.path.dirname(checkpoint_path)
    elif args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
        checkpoint_path = _latest_numeric_state_pt(checkpoint_dir)
    else:
        checkpoint_dir, checkpoint_path = resolve_diffuser_checkpoint(
            args.env_name,
            args.data_type,
            args.num_expert,
            horizon=args.horizon,
            logbase=args.logbase,
        )

    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No valid checkpoint in {checkpoint_dir}")
    print(f"[label] checkpoint_dir={checkpoint_dir}")
    print(f"[label] checkpoint_path={checkpoint_path}")

    print(f"Loading configs from {checkpoint_dir}")
    model_config = load_config(checkpoint_dir, "model_config.pkl")
    diffusion_config = load_config(checkpoint_dir, "diffusion_config.pkl")

    model_config._device = None
    diffusion_config._device = None
    model = model_config()
    if not hasattr(model, "encode_state_embedding"):
        raise AttributeError(
            "TemporalUnet is missing encode_state_embedding. "
            "Run with this repo on PYTHONPATH (do not use a stock pip diffuser)."
        )
    print(f"Instantiated model type: {type(model)}")

    diffuser = diffusion_config(model)
    load_diffuser_checkpoint(diffuser, checkpoint_path, device="cpu")
    diffuser.to(args.device)
    diffuser.eval()
    print(f"Using diffuser horizon={getattr(diffuser, 'horizon', None)} on {args.device}")

    mixed_dataset = create_mixed_dataset(
        env_name=args.env_name,
        data_type=args.data_type,
        num_expert_trajectories=args.num_expert,
    )

    offline_dataset, expert_dataset = split_offline_and_expert(mixed_dataset)
    print(f"Offline transitions: {len(offline_dataset['observations'])}")
    print(f"Expert transitions: {len(expert_dataset['observations'])}")

    reference = load_expert_reference(args.env_name, expert_data_path=args.expert_data_path)
    task_tag = f"{env_id}{'_mismatched' if args.mismatch else ''}-{args.num_expert}experts"
    labeled_dataset = score_offline_against_reference(
        mixed_dataset,
        reference,
        diffuser,
        temperature=args.score_temperature,
        device=args.device,
        k=args.k,
        task_description=task_tag,
        embed_batch_size=args.embed_batch_size,
    )
    np.savez_compressed(args.output_path, **labeled_dataset)
    print(f"Saved labeled dataset to {args.output_path}")

    mask = labeled_dataset["is_expert"].astype(bool)
    offline_rewards = labeled_dataset["rewards"][~mask]
    expert_rewards = labeled_dataset["rewards"][mask]
    
if __name__ == "__main__":
    main()
