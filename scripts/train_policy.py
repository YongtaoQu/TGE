import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run policy training.")
    parser.add_argument("--algo", type=str, default="rebrac", choices=["rebrac", "iql"])
    parser.add_argument("--env", type=str, default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expert_trajectories", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_timesteps", type=int, default=300000)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=20000)
    parser.add_argument("--labeled_dataset_path", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    return parser.parse_args()


def run_experiment(
    algo,
    env,
    seed,
    expert_trajectories,
    horizon,
    device,
    max_timesteps,
    labeled_dataset_path=None,
    n_episodes=100,
    eval_freq=20000,
    name=None,
    group=None,
):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_name = "rebrac.py" if algo == "rebrac" else "iql.py"
    script_path = os.path.join(root_dir, "policies", script_name)

    print("\n" + "=" * 60)
    print(f"STARTING {algo.upper()} TRAINING")
    print(f"Env: {env} | Seed: {seed} | Experts: {expert_trajectories} | Horizon: {horizon}")
    print(f"Device: {device} | Max steps: {max_timesteps}")
    if name:
        print(f"Wandb name: {name} | group: {group}")
    if labeled_dataset_path:
        print(f"Labeled dataset: {labeled_dataset_path}")
    print("=" * 60 + "\n")

    cmd = [
        sys.executable,
        script_path,
        "--env",
        env,
        "--seed",
        str(seed),
        "--expert_trajectories",
        str(expert_trajectories),
        "--horizon",
        str(horizon),
        "--device",
        device,
        "--max_timesteps",
        str(max_timesteps),
        "--n_episodes",
        str(n_episodes),
        "--eval_freq",
        str(eval_freq),
    ]
    if labeled_dataset_path:
        cmd.extend(["--labeled_dataset_path", labeled_dataset_path])
    if name:
        cmd.extend(["--name", name])
    if group:
        cmd.extend(["--group", group])

    try:
        subprocess.run(cmd, check=True)
        print(f"\n{algo.upper()} training on {env} (seed {seed}) finished successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{algo.upper()} training failed with return code {e.returncode}.")
        return False


def main():
    args = parse_args()
    ok = run_experiment(
        algo=args.algo,
        env=args.env,
        seed=args.seed,
        expert_trajectories=args.expert_trajectories,
        horizon=args.horizon,
        device=args.device,
        max_timesteps=args.max_timesteps,
        labeled_dataset_path=args.labeled_dataset_path,
        n_episodes=args.n_episodes,
        eval_freq=args.eval_freq,
        name=args.name,
        group=args.group,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
