import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

dfCPU = pd.read_csv("./dataCPU.csv")
dfGPU = pd.read_csv("./dataGPU.csv")

avg_dfGPU = dfGPU.groupby('seed')['totalRuntime'].mean().reset_index()
avg_dfCPU = dfCPU.groupby('seed')['totalRuntime'].mean().reset_index()

#dist_names = ["uniform square", "uniform disk", "gaussian"]
dist_names = ["uniform square", "uniform disk"]
for i, dist in enumerate(dist_names):
	dfGPU_distrib = dfGPU[ dfGPU["distribution"] == i ].groupby('npts')['totalRuntime'].mean().reset_index()
	dfCPU_distrib = dfCPU[ dfCPU["distribution"] == i ].groupby('npts')['totalRuntime'].mean().reset_index()
	
	nptsCPU = np.array(dfGPU_distrib["npts"])
	nptsGPU = np.array(dfGPU_distrib["npts"])

	timeGPU = np.array(dfGPU_distrib["totalRuntime"])
	timeCPU = np.array(dfCPU_distrib["totalRuntime"])

	speedup = timeGPU / timeCPU 

	plt.plot(nptsGPU, speedup, label=dist)
	plt.plot(nptsCPU, speedup, label=dist)

plt.xlabel("number of points")
plt.ylabel("speedup")
plt.legend()
plt.savefig("nptsVsSpeedup.png", dpi=200)
