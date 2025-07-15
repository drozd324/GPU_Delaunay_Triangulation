import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./data.csv")

# Average across seeds
metrics = ["prepForInsertTimeTot", "insertTimeTot", "flipTimeTot", "updatePtsTimeTot"]
avg_df = df.groupby("distribution")[metrics].mean()

print(avg_df)

# Normalize each row so values sum to 1 (i.e., percentage)
normalized = avg_df.div(avg_df.sum(axis=1), axis=0)

# Labels and styling
distributions = ["uniform square", "uniform disk", "gaussian"]
colors = ['forestgreen', 'gold', 'cornflowerblue', 'salmon']
hatches = ['..', '//', '\\\\', 'xx']
labels = ["prepForInsert", "insert", "flip", "updatePts"]

# Plotting
y = np.arange(len(distributions))
left = np.zeros(len(distributions))

plt.figure(figsize=(9, 4))
for i, (metric, color, hatch, label) in enumerate(zip(metrics, colors, hatches, labels)):
    values = normalized[metric].values
    plt.barh(y, values, left=left, color=color, edgecolor='black', hatch=hatch, label=label)
    left += values


plt.yticks(y, distributions)
plt.xlabel("Proportion of Total Time")
plt.xlim(0, 1)
plt.xticks(np.linspace(0, 1, 6), [f"{int(x*100)}%" for x in np.linspace(0, 1, 6)])
plt.legend(loc="upper right", ncol=2)
plt.tight_layout()

plt.savefig("timeDistrib.png", dpi=200)
plt.show()

