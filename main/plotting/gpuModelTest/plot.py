import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

deviceModels = ["NVIDIA GeForce RTX 2080 SUPER", "NVIDIA GeForce RTX 3090", "NVIDIA A100-PCIE-40GB", "NVIDIA A100-SXM4-80GB"]
# the following entry computes 1 / ((num of cores) * (frequency MHz)) for each model above. Source https://www.techpowerup.com 
normalization = [1/(3072*1650), 1/(10496*1395), 1/(6912*765), 1/(6912*1275)]

availableDeviceModels = [] 
times = []
for name in deviceModels:
	filepath = f"./data_{name}.csv"
	if not os.path.exists(filepath):
		continue

	df = pd.read_csv(filepath)
	avg_df = df.groupby("deviceName")[["totalRuntime"]].mean() # Average across seeds
	name_df = df[ df["deviceName"] == name]

	availableDeviceModels.append(name)
	times.append(*avg_df["totalRuntime"])

availableDeviceModels_sorted, times_sorted, normalization_sorted = zip(*sorted(zip(availableDeviceModels, times, normalization)))

availableDeviceModels_sorted = np.array(availableDeviceModels_sorted)
times_sorted = np.array(times_sorted)
normalization_sorted = np.array(normalization_sorted)

plt.bar(availableDeviceModels_sorted, times_sorted)
plt.plot(availableDeviceModels_sorted, times_sorted*normalization_sorted)

plt.xlabel("Device")
plt.ylabel("Normalized Time (seconds/((num cores) * (frequency mhz)) ")
plt.title("")

plt.savefig("gpuModelTest.png", dpi=200)
