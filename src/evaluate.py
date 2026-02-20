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


def make_env(env_name: str, seed: int, frame_stack: int = 4):
    """
    Create CarRacing environment using the SAME wrappers as training.
    """
    def _init():
        env = gym.make(env_name, render_mode="rgb_array")  # keep consistent + safe
        env = Monitor(env)
        env.reset(seed=seed)
        # extra reproducibility (when supported)
        try:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass
        return env

    env = DummyVecEnv([_init])
    env = VecTransposeImage(env)                # (H,W,C) -> (C,H,W)
    env = VecFrameStack(env, n_stack=frame_stack)  # stack frames
    return env


def eval_one_seed(env_name: str, model_path: str, seed: int, n_eval_episodes: int,
                  success_threshold: float, frame_stack: int = 4):
    """
    Evaluate a model for a given seed and return summary metrics.
    """
    env = make_env(env_name, seed, frame_stack=frame_stack)
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
        "frame_stack": int(frame_stack),
        "env_name": str(env_name),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Path to SB3 PPO .zip model")
    p.add_argument("--out_dir", type=str, default="reports", help="Where to save CSV/JSON results")

    # Optional: load defaults from YAML config
    p.add_argument("--config", type=str, default=None, help="Optional YAML config for eval defaults")

    # CLI defaults (used if no config)
    p.add_argument("--n_eval_episodes", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--success_threshold", type=float, default=200.0)
    p.add_argument("--frame_stack", type=int, default=4)

    args = p.parse_args()
    env_name = "CarRacing-v2"

    # If config provided, override eval settings from YAML
    if args.config is not None:
        cfg = yaml.safe_load(Path(args.config).read_text())

        env_cfg = cfg.get("environment", {})
        env_name = env_cfg.get("name", env_name)

        tr_cfg = cfg.get("training", {})
        args.frame_stack = int(tr_cfg.get("frame_stack", args.frame_stack))

        eval_cfg = cfg.get("evaluation", {})
        args.n_eval_episodes = int(eval_cfg.get("n_eval_episodes", args.n_eval_episodes))
        args.seeds = list(eval_cfg.get("seeds", args.seeds))
        args.success_threshold = float(eval_cfg.get("success_threshold", args.success_threshold))

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for seed in args.seeds:
        r = eval_one_seed(
            env_name=env_name,
            model_path=args.model_path,
            seed=seed,
            n_eval_episodes=args.n_eval_episodes,
            success_threshold=args.success_threshold,
            frame_stack=args.frame_stack,
        )
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
                "frame_stack", "env_name",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", json_path)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
