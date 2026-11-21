# TGE

Code for *Imitation from Observations with Trajectory-Level Generative Embeddings* ([NeurIPS 2026](https://neurips.cc/virtual/2026/loc/atlanta/poster/151602)).

Pipeline per setting: **train diffuser → label rewards → train ReBRAC**.

For code execution or reproduction issues, contact [yongtao@unc.edu](mailto:yongtao@unc.edu).

## Setup

```bash
cd /path/to/TGE-Project
export PYTHONPATH=$PWD:$PYTHONPATH   # required (uses this repo's diffuser/)
pip install -r requirements.txt
```

## Settings

| Domain | `env_name` | `data_type` | `N` |
|--------|------------|-------------|-----|
| Locomotion | `hopper` / `walker2d` / `halfcheetah` / `ant` | `medium` | 0, 30, 200 |
| Locomotion | same | `random` | 30, 200 |
| Locomotion | same | `random-medium` | 0 |
| Adroit | `pen` / `door` / `hammer` / `relocate` | `cloned` / `human` | 30 |
| Kitchen | `kitchen` | `partial` / `mixed` | 30 |
| Mismatch | `halfcheetah` | Short expert | 200 |
| Mismatch | `ant` | Disabled expert | 200 |


## Run one setting

Example: `walker2d` + `medium` + `N=30` + `H=32`.

```bash
# 1) diffuser
python scripts/train_diffuser.py \
  --env_name walker2d \
  --data_type medium \
  --num_expert_trajectories 30 \
  --horizon 32

# 2) TGE label
python scripts/label_rewards.py \
  --env_name walker2d \
  --data_type medium \
  --num_expert 30 \
  --horizon 32

# 3) ReBRAC
python scripts/train_policy.py \
  --algo rebrac \
  --env walker2d-medium-v2 \
  --expert_trajectories 30 \
  --horizon 32 \
  --max_timesteps 300000
```

Adroit / kitchen: change `env_name`, `data_type`, `N`, and set `--env` to the matching gym id (`*-v0`).

## Mismatched experts

```bash
# HalfCheetah-Short
python scripts/train_diffuser.py \
  --env_name halfcheetah --data_type random --num_expert_trajectories 200 --horizon 32

python scripts/label_rewards.py \
  --env_name halfcheetah --data_type random --num_expert 200 --horizon 32 \
  --expert_data_path expert_data/halfcheetah_short_expert.pkl --mismatch

python scripts/train_policy.py \
  --algo rebrac --env halfcheetah-random-v2 --expert_trajectories 200 --horizon 32 \
  --labeled_dataset_path labeled_dataset/H32/halfcheetah-random-v2_mismatched_N200/labeled_dataset.npz

# Ant-Disabled
python scripts/train_diffuser.py \
  --env_name ant --data_type random --num_expert_trajectories 200 --horizon 32

python scripts/label_rewards.py \
  --env_name ant --data_type random --num_expert 200 --horizon 32 \
  --expert_data_path expert_data/ant_disabled_expert.pkl --mismatch

python scripts/train_policy.py \
  --algo rebrac --env ant-random-v2 --expert_trajectories 200 --horizon 32 \
  --labeled_dataset_path labeled_dataset/H32/ant-random-v2_mismatched_N200/labeled_dataset.npz
```

## Layout

| Path | Role |
|------|------|
| `diffuser/` | Model + `config/` + offline mixing |
| `expert_data/` | Expert reference `.pkl` |
| `policies/` | ReBRAC / IQL |
| `scripts/` | `train_diffuser.py` / `label_rewards.py` / `train_policy.py` |
