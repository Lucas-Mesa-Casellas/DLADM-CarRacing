from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to eval_results.csv")
    parser.add_argument("--out", type=str, required=True, help="Output folder for plots")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).sort_values("seed")

    # Print summary (nice for report copy/paste)
    overall_mean = df["reward_mean"].mean()
    overall_std = df["reward_mean"].std(ddof=1) if len(df) > 1 else 0.0

    print("\nGeneralisation summary:")
    print(df[["seed", "reward_mean", "reward_std", "len_mean", "success_rate"]].to_string(index=False))
    print(f"\nOverall reward_mean across seeds: {overall_mean:.2f} ± {overall_std:.2f}")
    print(f"Overall success_rate across seeds: {df['success_rate'].mean():.2f}")

    # Plot 1: reward mean ± std per seed
    plt.figure()
    plt.errorbar(df["seed"], df["reward_mean"], yerr=df["reward_std"], fmt="o", capsize=4)
    plt.xlabel("Evaluation seed (unseen)")
    plt.ylabel("Episode reward (mean ± std)")
    plt.title("PPO generalisation on CarRacing-v3")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = out_dir / "reward_by_seed.png"
    plt.savefig(p1, dpi=200)
    print("Saved plot:", p1)

    # Plot 2: success rate per seed
    plt.figure()
    plt.plot(df["seed"], df["success_rate"], marker="o")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Evaluation seed (unseen)")
    plt.ylabel("Success rate")
    plt.title("Success rate by seed")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = out_dir / "success_rate_by_seed.png"
    plt.savefig(p2, dpi=200)
    print("Saved plot:", p2)


if __name__ == "__main__":
    main()
