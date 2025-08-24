import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./data.csv")

for i, numpoints in enumerate([100, 1000, 10000]):
	subdf = df[ df["npts"] == numpoints ]
	print(subdf)

	plt.subplot(3, 1, i+1)
	
	# Average across seeds
	metrics = ["prepForInsertTimeTot", "insertTimeTot", "flipTimeTot", "updatePtsTimeTot"]
	avg_subdf = subdf.groupby("distribution")[metrics].mean()

	#print(avg_subdf)

	# Normalize each row so values sum to 1 (i.e., percentage)
	normalized = avg_subdf.div(avg_subdf.sum(axis=1), axis=0)

	# Labels and styling
	distributions = ["Uniform", "Clustered center", "Clustered boundary", "Gaussian"]
	colors = ['forestgreen', 'gold', 'cornflowerblue', 'salmon']
	hatches = ['..', '//', '\\\\', 'xx']
	labels = ["prepForInsert", "insert", "flip", "updatePointLocations"]

	# Plotting
	y = np.arange(len(distributions))
	left = np.zeros(len(distributions))

	for i, (metric, color, hatch, label) in enumerate(zip(metrics, colors, hatches, labels)):
		values = normalized[metric].values
		plt.barh(y, values, left=left, color=color, edgecolor='black', hatch=hatch, label=label)
		left += values


	plt.yticks(y, distributions)
	plt.xlabel("Proportion of Total Time")
	plt.xlim(0, 1)
	plt.xticks(np.linspace(0, 1, 6), [f"{int(x*100)}%" for x in np.linspace(0, 1, 6)])
	#plt.legend(loc="upper center", ncol=2)
	plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2)

#plt.figure(figsize=(9, 4))
plt.tight_layout()

plt.savefig("timeDistrib.png", dpi=200)
