import logging
import os

import d4rl
import gym
import numpy as np

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
logging.getLogger("d4rl").setLevel(logging.WARNING)


def create_mixed_dataset(
    env_name: str = "walker2d",
    data_type: str = "medium",
    num_expert_trajectories: int = 200,
):
    """Mix a D4RL offline split with N expert trajectories."""
    print(f"Creating mixed dataset for {env_name} (type: {data_type})...")

    if data_type == "random-medium":
        env_random = gym.make(f"{env_name}-random-v2")
        env_medium = gym.make(f"{env_name}-medium-v2")
        ds_random = d4rl.qlearning_dataset(env_random)
        ds_medium = d4rl.qlearning_dataset(env_medium)
        n_half = 500000
        combined = {
            "observations": np.concatenate(
                (ds_random["observations"][:n_half], ds_medium["observations"][:n_half]), axis=0
            ),
            "actions": np.concatenate(
                (ds_random["actions"][:n_half], ds_medium["actions"][:n_half]), axis=0
            ),
            "rewards": np.concatenate(
                (ds_random["rewards"][:n_half], ds_medium["rewards"][:n_half]), axis=0
            ),
            "terminals": np.concatenate(
                (ds_random["terminals"][:n_half], ds_medium["terminals"][:n_half]), axis=0
            ),
            "next_observations": np.concatenate(
                (ds_random["next_observations"][:n_half], ds_medium["next_observations"][:n_half]),
                axis=0,
            ),
            "is_expert": np.zeros(n_half * 2, dtype=np.float32),
        }
        data = _mark_dones(combined, env_name)
        print(f"Done! Total transitions: {len(data['observations'])} (500k random + 500k medium)")
        return data

    if data_type == "random":
        offline_env_name = f"{env_name}-random-v2"
    elif data_type == "medium":
        offline_env_name = f"{env_name}-medium-v2"
    elif data_type == "human":
        offline_env_name = f"{env_name}-human-v0"
    elif data_type == "cloned":
        offline_env_name = f"{env_name}-cloned-v0"
    elif data_type == "partial":
        offline_env_name = f"{env_name}-partial-v0"
    elif data_type == "mixed":
        offline_env_name = f"{env_name}-mixed-v0"
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    if env_name == "kitchen":
        expert_env_name = f"{env_name}-complete-v0"
    elif env_name in ["door", "pen", "hammer", "relocate"]:
        expert_env_name = f"{env_name}-expert-v0"
    else:
        expert_env_name = f"{env_name}-expert-v2"

    data = _combine_offline_and_expert_data(
        offline_env=gym.make(offline_env_name),
        expert_env=gym.make(expert_env_name),
        expert_trajectories=num_expert_trajectories,
        env_name=env_name,
    )
    print(
        f"Done! Total transitions: {len(data['observations'])}, "
        f"Expert transitions: {int(data['is_expert'].sum())}"
    )
    return data


def _combine_offline_and_expert_data(
    offline_env: gym.Env,
    expert_env: gym.Env,
    expert_trajectories: int,
    env_name: str,
):
    dataset = d4rl.qlearning_dataset(offline_env)
    expert_dataset = d4rl.qlearning_dataset(expert_env)

    if "kitchen" in env_name:
        max_steps = 280
    elif "pen" in env_name:
        max_steps = 100
    elif "door" in env_name or "hammer" in env_name or "relocate" in env_name:
        max_steps = 200
    else:
        max_steps = 1000

    if expert_trajectories <= 0:
        print("[ Dataset ] num_expert=0 → offline-only mixed buffer")
        combined_dataset = {
            "observations": dataset["observations"],
            "actions": dataset["actions"],
            "rewards": dataset["rewards"],
            "terminals": dataset["terminals"],
            "next_observations": dataset["next_observations"],
            "is_expert": np.zeros(dataset["observations"].shape[0], dtype=np.float32),
        }
        return _mark_dones(combined_dataset, env_name)

    expert_transitions = 0
    traj_count = 0
    episode_step = 0
    for i in range(expert_dataset["observations"].shape[0]):
        episode_step += 1
        if episode_step == max_steps or expert_dataset["terminals"][i]:
            traj_count += 1
            episode_step = 0
        if traj_count == expert_trajectories:
            expert_transitions = i + 1
            break

    if traj_count < expert_trajectories:
        expert_transitions = expert_dataset["observations"].shape[0]
        print(
            f"[ Dataset ] Warning: expert split has only {traj_count} trajectories "
            f"(requested {expert_trajectories}); using all {expert_transitions} expert transitions."
        )
    else:
        print(f"Expert data: {expert_trajectories} trajectories, {expert_transitions} transitions")

    combined_dataset = {
        "observations": np.concatenate(
            (dataset["observations"], expert_dataset["observations"][:expert_transitions]), axis=0
        ),
        "actions": np.concatenate(
            (dataset["actions"], expert_dataset["actions"][:expert_transitions]), axis=0
        ),
        "rewards": np.concatenate(
            (dataset["rewards"], expert_dataset["rewards"][:expert_transitions]), axis=0
        ),
        "terminals": np.concatenate(
            (dataset["terminals"], expert_dataset["terminals"][:expert_transitions]), axis=0
        ),
        "next_observations": np.concatenate(
            (dataset["next_observations"], expert_dataset["next_observations"][:expert_transitions]),
            axis=0,
        ),
        "is_expert": np.concatenate(
            (np.zeros(dataset["observations"].shape[0]), np.ones(expert_transitions)), axis=0
        ),
    }
    return _mark_dones(combined_dataset, env_name)


def _mark_dones(combined_dataset, env_name):
    num_total = len(combined_dataset["rewards"])
    dones = np.zeros(num_total, dtype=np.float32)

    if "kitchen" in env_name:
        env_max_steps = 280
    elif "pen" in env_name:
        env_max_steps = 100
    elif "door" in env_name or "hammer" in env_name or "relocate" in env_name:
        env_max_steps = 200
    else:
        env_max_steps = 1000

    episode_step = 0
    for i in range(num_total):
        episode_step += 1
        is_terminal = combined_dataset["terminals"][i] == 1.0
        is_timeout = episode_step >= env_max_steps
        is_jump = False
        if i < num_total - 1:
            dist = np.linalg.norm(
                combined_dataset["observations"][i + 1] - combined_dataset["next_observations"][i]
            )
            if dist > 1e-6:
                is_jump = True
        if is_terminal or is_timeout or is_jump:
            dones[i] = 1.0
            episode_step = 0

    return {
        "observations": combined_dataset["observations"].astype(np.float32),
        "actions": combined_dataset["actions"].astype(np.float32),
        "rewards": combined_dataset["rewards"].astype(np.float32),
        "terminals": combined_dataset["terminals"].astype(np.float32),
        "next_observations": combined_dataset["next_observations"].astype(np.float32),
        "is_expert": combined_dataset["is_expert"].astype(np.float32),
        "dones": dones.astype(np.float32),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        env_name = sys.argv[1]
        data_type = sys.argv[2] if len(sys.argv) > 2 else "human"
        num_expert = int(sys.argv[3]) if len(sys.argv) > 3 else 200
        data = create_mixed_dataset(env_name, data_type, num_expert)
        print(f"Dataset created with {len(data['observations'])} transitions")
    else:
        print("Usage: python create_offline_dataset.py <env_name> [data_type] [num_expert_trajectories]")
        print("Example: python create_offline_dataset.py hammer human 200")
