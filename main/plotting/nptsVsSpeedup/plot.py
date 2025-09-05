import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

dfCPU = pd.read_csv("./dataCPU.csv")
dfGPU = pd.read_csv("./dataGPU.csv")

dist_names = ["Uniform", "Clustered center", "Clustered boudary", "Gaussian"]
for i, dist in enumerate(dist_names):
	dfGPU_distrib = dfGPU[ dfGPU["distribution"] == i ].groupby('npts')['totalRuntime'].mean().reset_index()
	dfCPU_distrib = dfCPU[ dfCPU["distribution"] == i ].groupby('npts')['totalRuntime'].mean().reset_index()
	
	nptsGPU = np.array(dfGPU_distrib["npts"])
	nptsCPU = np.array(dfCPU_distrib["npts"])

	timeGPU = np.array(dfGPU_distrib["totalRuntime"])
	timeCPU = np.array(dfCPU_distrib["totalRuntime"])

	speedup = timeCPU / timeGPU[:len(timeCPU)]   

	#plt.plot(nptsGPU, speedup, label=dist)
	#plt.plot(nptsCPU, speedup, label=dist)
	plt.loglog(nptsCPU, speedup, label=dist)
	#plt.semilogx(nptsCPU, speedup, label=dist)

plt.xlabel("Number of points")
plt.ylabel(r"Speedup ($\frac{time CPU}{time GPU}$)")
#plt.xscale("log", base=2)
#plt.yscale("log", base=2)

plt.legend()
plt.savefig("nptsVsSpeedup.png", dpi=200)
