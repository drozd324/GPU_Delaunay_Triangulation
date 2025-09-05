import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

deviceModels = [
    #"NVIDIA GeForce GTX 1080 Ti",
    "NVIDIA GeForce RTX 2080 SUPER",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4060 Ti",
    "NVIDIA A100-PCIE-40GB",
    "NVIDIA A100-SXM4-80GB",
	"Tesla_V100-PCIE-16GB"
]

# the following entry computes 1 / ((num of cores) * (clock frequency MHz)) for each model above. Source https://www.techpowerup.com 
normalization = [
    #1/(3584*1481),
    1/(3072*1650),
    1/(10496*1395),
    1/(4352*2310),
    1/(6912*765),
    1/(6912*1275),
    1/(5120*1245)
]

availableDeviceModels = [] 
availableNormalization = [] 
times = []
for i, name in enumerate(deviceModels):
    filepath = f"./data_{name}.csv"
    if not os.path.exists(filepath):
        continue

    df = pd.read_csv(filepath)
    avg_df = df.groupby("deviceName")[["totalRuntime"]].mean() # Average across seeds

    availableDeviceModels.append(name)
    times.append(*avg_df["totalRuntime"])
    availableNormalization.append(normalization[i])

times_sorted, availableDeviceModels_sorted, availableNormalization_sorted = zip(
    *sorted(zip(times, availableDeviceModels, normalization), reverse=True)
)

times_sorted = np.array(times_sorted)
availableNormalization_sorted = np.array(availableNormalization_sorted)

normalized_times = times_sorted * availableNormalization_sorted

fig, ax1 = plt.subplots(figsize=(12,6))
ax2 = ax1.twinx()

x = np.arange(len(availableDeviceModels_sorted))  
width = 0.4

bars1 = ax1.bar(x - width/2, times_sorted, width, label="Total time",
                edgecolor="black", color="cornflowerblue", hatch="//")

bars2 = ax2.bar(x + width/2, normalized_times, width, label="Normalized time",
                edgecolor="black", color="salmon", hatch="..")

ax1.set_xlabel("Device")
ax1.set_ylabel("Total Time (s)")
ax2.set_ylabel("Normalized Time (s / (#cores * MHz))")

ax1.set_xticks(x)
ax1.set_xticklabels(availableDeviceModels_sorted, rotation=5, ha="right")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.tight_layout()
plt.savefig("gpuModelTest.png", dpi=200)
