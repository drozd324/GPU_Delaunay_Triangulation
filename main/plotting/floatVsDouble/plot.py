import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

dfFloat = pd.read_csv("./dataFloat.csv")
dfDouble = pd.read_csv("./dataDouble.csv")

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
dist_names = ["uniform", "clustered center", "clustered boundary", "gaussian"]
#plt.figure(figsize=(10, 4))
nptsFloat = []
for i, dist in enumerate(dist_names):
	dfDouble_distrib = dfDouble[ dfDouble["distribution"] == i ].groupby('npts')['totalRuntime'].mean().reset_index()
	dfFloat_distrib = dfFloat[ dfFloat["distribution"] == i ].groupby('npts')['totalRuntime'].mean().reset_index()
	
	nptsDouble = np.array(dfDouble_distrib["npts"])
	nptsFloat = np.array(dfFloat_distrib["npts"])

	timeDouble = np.array(dfDouble_distrib["totalRuntime"])
	timeFloat = np.array(dfFloat_distrib["totalRuntime"])

	#plt.subplot(1, 2, 1)
	plt.plot(nptsDouble, timeDouble, label=f"{dist} double", linestyle="--", color=colors[i])
	plt.plot(nptsFloat,  timeFloat, label=f"{dist} float"  , linestyle="-" , color=colors[i])

#	plt.subplot(1, 2, 2)
#	plt.loglog(nptsDouble, timeDouble, label=f"{dist} double", linestyle="--", color=colors[i])
#	plt.loglog(nptsFloat,  timeFloat, label=f"{dist} float"  , linestyle="-" , color=colors[i])

#plt.subplot(1, 2, 1)
#plt.plot(nptsFloat, nptsFloat**2, label=r"$x^2$")
plt.xlabel("Number of points")
plt.ylabel("Time (s)")
plt.legend()

#plt.subplot(1, 2, 2)
#plt.xlabel("Number of points")
#plt.ylabel("Time (s)")
#plt.legend()

plt.savefig("floatVsDouble.png", dpi=200)
