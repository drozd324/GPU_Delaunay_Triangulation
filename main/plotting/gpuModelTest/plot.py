import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

deviceModels = [
	#"NVIDIA GeForce GTX 1080 Ti",
	"NVIDIA GeForce RTX 2080 SUPER",
	"NVIDIA GeForce RTX 3090",
	"NVIDIA A100-PCIE-40GB",
	"NVIDIA A100-SXM4-80GB"
]

# the following entry computes 1 / ((num of cores) * (clock frequency MHz)) for each model above. Source https://www.techpowerup.com 
normalization = [
	#1/(3584*1481),
	1/(3072*1650),
	1/(10496*1395),
	1/(6912*765),
	1/(6912*1275)
]

# the following entry computes 1 / ((num of cores) * (clock frequency memory MHz)) for each model above. Source https://www.techpowerup.com 
#normalization_mem = [1/(35841*), 1/(3072*), 1/(10496*), 1/(6912*), 1/(6912*)]

availableDeviceModels = [] 
availableNormalization = [] 
times = []
for i, name in enumerate(deviceModels):
	filepath = f"./data_{name}.csv"
	if not os.path.exists(filepath):
		continue

	df = pd.read_csv(filepath)
	avg_df = df.groupby("deviceName")[["totalRuntime"]].mean() # Average across seeds
	name_df = df[ df["deviceName"] == name]

	availableDeviceModels.append(name)
	times.append(*avg_df["totalRuntime"])
	availableNormalization.append(normalization[i])

times_sorted, availableDeviceModels_sorted, availableNormalization_sorted = zip(*sorted(zip(times, availableDeviceModels, normalization), reverse=True))

availableDeviceModels_sorted = availableDeviceModels_sorted
times_sorted = np.array(times_sorted)
availableNormalization_sorted = np.array(availableNormalization_sorted)

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(12,5))

# Bar chart on left y-axis
bars = ax1.bar(availableDeviceModels_sorted, times_sorted, label="Total time", edgecolor="black", color="cornflowerblue")
for bar in bars:
	bar.set_hatch("//")

ax1.set_xlabel("Device")
ax1.set_ylabel("Total Time (s)")
ax1.tick_params(axis='y')

#colors = ['forestgreen', 'gold', 'cornflowerblue', 'salmon']
# Create a second y-axis
ax2 = ax1.twinx()
ax2.plot(availableDeviceModels_sorted, times_sorted*availableNormalization_sorted, color='red', marker='o', linestyle='-', label="Normalized time")
ax2.set_ylabel("Normalized Time (s/#cores * MHz)")
ax2.tick_params(axis='y')

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2 )

plt.savefig("gpuModelTest.png", dpi=200)
