import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("algo_logs/eval_rollouts.csv")

for algo in df["algo"].unique():
    sub = df[df["algo"] == algo]

    grouped = sub.groupby("timestep")["reward"].mean()

    plt.plot(grouped.index, grouped.values, label=algo)

plt.xlabel("Timestep")
plt.ylabel("Average Reward")
plt.grid('on')
plt.legend()
plt.show()
