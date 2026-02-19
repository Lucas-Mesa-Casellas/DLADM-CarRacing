import pandas as pd

base = pd.read_csv("reports/base/eval_results.csv")
lr = pd.read_csv("reports/lr_1e4/eval_results.csv")

print("\n=== MODEL COMPARISON ===")

print("\nBASE mean reward:", base["reward_mean"].mean())
print("LR_1e4 mean reward:", lr["reward_mean"].mean())

print("\nBASE success rate:", base["success_rate"].mean())
print("LR_1e4 success rate:", lr["success_rate"].mean())
