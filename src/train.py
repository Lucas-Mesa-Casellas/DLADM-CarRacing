from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import yaml
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from src.utils import set_global_seed, collect_run_info


def make_env(seed: int):
    """Create a single CarRacing env with monitoring + deterministic reset seed."""
    def _init():
        env = gym.make("CarRacing-v3")
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to YAML config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text())
    seed = int(cfg.get("seed", 42))

    set_global_seed(seed)
    print(collect_run_info(seed))

    # Experiment naming (prevents overwriting logs/models)
    exp_name = cfg.get("experiment_name", cfg_path.stem)

    # Paths
    log_dir = Path("logs/ppo") / exp_name
    model_dir = Path("results/models")
    run_cfg_dir = Path("reports") / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    run_cfg_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact config used (reproducibility for report/markers)
    shutil.copyfile(cfg_path, run_cfg_dir / "config_used.yaml")

    # Env
    env = DummyVecEnv([make_env(seed)])
    env = VecFrameStack(env, n_stack=int(cfg["training"].get("frame_stack", 4)))

    # Training params
    tr = cfg["training"]
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


if __name__ == "__main__":
    main()
