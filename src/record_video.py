import os
import argparse

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor


def make_env(env_name: str, seed: int, video_folder: str, name_prefix: str, frame_stack: int = 4):
    def _init():
        env = gym.make(env_name, render_mode="rgb_array")
        env = Monitor(env)
        env = RecordVideo(
            env,
            video_folder=video_folder,
            name_prefix=name_prefix,
            episode_trigger=lambda ep: True,
        )
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass
        return env

    env = DummyVecEnv([_init])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=frame_stack)
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)  # SB3 loads .zip automatically
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="videos")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--env_name", type=str, default="CarRacing-v2")
    p.add_argument("--frame_stack", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    env = make_env(
        env_name=args.env_name,
        seed=args.seed,
        video_folder=args.out_dir,
        name_prefix=f"ppo_seed{args.seed}",
        frame_stack=args.frame_stack,
    )
    model = PPO.load(args.model_path)

    obs = env.reset()
    ep = 0
    while ep < args.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        if bool(dones[0]):
            ep += 1
            obs = env.reset()

    env.close()
    print(f"Saved video(s) to: {args.out_dir}")


if __name__ == "__main__":
    main()
