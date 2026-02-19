import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RecordVideo


def make_env(seed: int, video_folder: str, name_prefix: str):
    def _init():
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        env = Monitor(env)
        env = RecordVideo(env, video_folder=video_folder, name_prefix=name_prefix, episode_trigger=lambda ep: True)
        env.reset(seed=seed)
        return env

    env = DummyVecEnv([_init])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)  # without .zip is fine
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="videos")
    p.add_argument("--episodes", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    env = make_env(args.seed, args.out_dir, name_prefix=f"ppo_seed{args.seed}")
    model = PPO.load(args.model_path)

    # Run N episodes
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
