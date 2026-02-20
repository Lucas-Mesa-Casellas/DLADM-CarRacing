import os
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import yaml

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor


def make_env(env_name: str, seed: int):
    """
    Create CarRacing environment using the SAME wrappers as training.
    """
    def _init():
        env = gym.make(env_name)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    env = DummyVecEnv([_init])
    env = VecTransposeImage(env)        # (H,W,C) -> (C,H,W)
    env = VecFrameStack(env, n_stack=4) # 4 frames -> 12 channels
    return env


def eval_one_seed(env_name: str, model_path: str, seed: int, n_eval_episodes: int, success_threshold: float):
    """
    Evaluate a model for a given seed and return summary metrics.
    """
    env = make_env(env_name, seed)
    model = PPO.load(model_path)

    ep_rewards, ep_lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        return_episode_rewards=True,
    )
    env.close()

    ep_rewards = np.array(ep_rewards, dtype=float)
    ep_lengths = np.array(ep_lengths, dtype=float)

    success_rate = float(np.mean(ep_rewards >= success_threshold))

    return {
        "seed": int(seed),
        "n_eval_episodes": int(n_eval_episodes),
        "reward_mean": float(ep_rewards.mean()),
        "reward_std": float(ep_rewards.std(ddof=1)) if len(ep_rewards) > 1 else 0.0,
        "len_mean": float(ep_lengths.mean()),
        "len_std": float(ep_lengths.std(ddof=1)) if len(ep_lengths) > 1 else 0.0,
        "success_rate": success_rate,
        "success_threshold": float(success_threshold),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Path to SB3 PPO .zip model")
    p.add_argument("--out_dir", type=str, default="reports", help="Where to save CSV/JSON results")

    # Optional: load defaults from YAML config
    p.add_argument("--config", type=str, default=None, help="Optional YAML config for eval defaults")

    # CLI defaults (used if no config)
    p.add_argument("--n_eval_episodes", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--success_threshold", type=float, default=200.0)

    args = p.parse_args()

    # Defaults
    env_name = "CarRacing-v3"

    # If config provided, override eval settings from YAML
    if args.config is not None:
        cfg = yaml.safe_load(Path(args.config).read_text())

        env_cfg = cfg.get("environment", {})
        env_name = env_cfg.get("name", env_name)

        eval_cfg = cfg.get("evaluation", {})
        args.n_eval_episodes = int(eval_cfg.get("n_eval_episodes", args.n_eval_episodes))
        args.seeds = list(eval_cfg.get("seeds", args.seeds))
        args.success_threshold = float(eval_cfg.get("success_threshold", args.success_threshold))

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for seed in args.seeds:
        r = eval_one_seed(env_name, args.model_path, seed, args.n_eval_episodes, args.success_threshold)
        rows.append(r)
        print(
            f"[seed={seed}] reward_mean={r['reward_mean']:.1f}±{r['reward_std']:.1f} "
            f"len_mean={r['len_mean']:.0f} success_rate={r['success_rate']:.2f}"
        )

    # Save JSON
    json_path = os.path.join(args.out_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": rows}, f, indent=2)

    # Save CSV
    csv_path = os.path.join(args.out_dir, "eval_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed", "n_eval_episodes",
                "reward_mean", "reward_std",
                "len_mean", "len_std",
                "success_rate", "success_threshold",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", json_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
