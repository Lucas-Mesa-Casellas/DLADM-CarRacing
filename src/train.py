from __future__ import annotations

from pathlib import Path
import yaml
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from src.utils import set_global_seed, collect_run_info


def make_env(seed: int):
    def _init():
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    # Load config
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text())
    seed = int(cfg["seed"])

    set_global_seed(seed)
    print(collect_run_info(seed))

    # Paths
    log_dir = Path("logs/ppo")
    model_dir = Path("results/models")
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Env
    env = DummyVecEnv([make_env(seed)])
    env = VecFrameStack(env, n_stack=4)

    # Model
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=float(cfg["training"]["learning_rate"]),
        n_steps=int(cfg["training"]["n_steps"]),
        batch_size=int(cfg["training"]["batch_size"]),
        gamma=float(cfg["training"]["gamma"]),
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=seed,
    )

    # Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(model_dir),
        name_prefix="ppo_carracing",
    )

    total_timesteps = int(cfg["training"]["total_timesteps"])
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    model.save(str(model_dir / "ppo_carracing_final"))
    print("Saved final model to", model_dir / "ppo_carracing_final.zip")


if __name__ == "__main__":
    main()
