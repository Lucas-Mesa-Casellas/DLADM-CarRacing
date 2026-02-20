from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import gymnasium as gym
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from src.utils import set_global_seed, collect_run_info
from src.wrappers import ActionDTypeWrapper


def make_env(env_name: str, seed: int):
    """
    Create a single Gymnasium env with monitoring + deterministic reset seed.
    Wrapped inside DummyVecEnv in main().
    """
    def _init():
        env = gym.make(env_name)
        env = ActionDTypeWrapper(env)  # Fix Box2D float32 action dtype crash
        env = Monitor(env)
        env.reset(seed=seed)
        # Extra reproducibility (best-effort)
        try:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="SB3 device: auto/cpu/cuda",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text())
    seed = int(cfg.get("seed", 42))

    # Read environment from config
    env_cfg = cfg.get("environment", {})
    env_name = env_cfg.get("name", "CarRacing-v2")

    # Reproducibility
    set_global_seed(seed)
    run_info = collect_run_info(seed)
    print(run_info)

    # Experiment naming (prevents overwriting logs/models)
    exp_name = cfg.get("experiment_name", cfg_path.stem)

    # Paths
    log_dir = Path("logs/ppo") / exp_name
    model_dir = Path("results/models") / exp_name
    run_cfg_dir = Path("reports") / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_cfg_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact config used
    shutil.copyfile(cfg_path, run_cfg_dir / "config_used.yaml")

    # Save run info (versions + cuda availability)
    (run_cfg_dir / "run_info.txt").write_text(str(run_info), encoding="utf-8")

    # Env (MATCH evaluate.py / record_video.py wrappers)
    tr = cfg["training"]
    frame_stack = int(tr.get("frame_stack", 4))

    env = DummyVecEnv([make_env(env_name, seed)])
    env = VecTransposeImage(env)  # (H,W,C) -> (C,H,W) for CnnPolicy
    env = VecFrameStack(env, n_stack=frame_stack)

    # Training params
    total_timesteps = int(tr["total_timesteps"])
    checkpoint_freq = int(tr.get("checkpoint_freq", 50_000))

    # Model
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=float(tr["learning_rate"]),
        n_steps=int(tr["n_steps"]),
        batch_size=int(tr["batch_size"]),
        gamma=float(tr["gamma"]),
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=seed,
        device=args.device,
    )

    # Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(model_dir),
        name_prefix=f"ppo_{exp_name}",
    )

    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Final save (SB3 adds .zip)
    final_name = f"ppo_{exp_name}_final"
    model.save(str(model_dir / final_name))
    print("Saved final model to", model_dir / f"{final_name}.zip")

    env.close()


if __name__ == "__main__":
    main()
