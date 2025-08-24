import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./data.csv")

npoints_list = [100, 1000, 10000]

fig, axes = plt.subplots(len(npoints_list), 1, figsize=(10, 9), sharex=True)

for i, numpoints in enumerate(npoints_list):
	subdf = df[df["npts"] == numpoints]

	metrics = ["prepForInsertTimeTot", "insertTimeTot", "flipTimeTot", "updatePtsTimeTot"]
	avg_subdf = subdf.groupby("distribution")[metrics].mean()

	normalized = avg_subdf.div(avg_subdf.sum(axis=1), axis=0)

	distributions = ["Uniform", "Clustered center", "Clustered boundary", "Gaussian"]
	colors = ['forestgreen', 'gold', 'cornflowerblue', 'salmon']
	hatches = ['..', '//', '\\\\', 'xx']
	labels = ["prepForInsert", "insert", "flip", "updatePointLocations"]

	ax = axes[i]
	y = np.arange(len(distributions))
	left = np.zeros(len(distributions))

	for j, (metric, color, hatch, label) in enumerate(zip(metrics, colors, hatches, labels)):
		values = normalized[metric].values
		ax.barh(y, values, left=left, color=color, edgecolor='black', hatch=hatch, label=label if i == 0 else "")
		left += values

	ax.set_yticks(y)
	ax.set_yticklabels(distributions)
	ax.set_xlim(0, 1)
	ax.set_xticks(np.linspace(0, 1, 6))
	ax.set_xticklabels([f"{int(x*100)}%" for x in np.linspace(0, 1, 6)])
	if i == 2:
		ax.set_xlabel("Proportion of Total Time")
	exp = int(np.log10(numpoints))
	ax.set_title(f"Number of Points = $10^{exp}$")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 1.00), ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("timeDistrib.png", dpi=200)

