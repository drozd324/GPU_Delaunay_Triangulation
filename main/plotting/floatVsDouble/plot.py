import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

dfFloat = pd.read_csv("./dataFloat.csv")
dfDouble = pd.read_csv("./dataDouble.csv")

dist_names = ["uniform", "clustered center", "clustered boudary", "gaussian"]
for i, dist in enumerate(dist_names):
	dfDouble_distrib = dfDouble[ dfDouble["distribution"] == i ].groupby('npts')['totalRuntime'].mean().reset_index()
	dfFloat_distrib = dfFloat[ dfFloat["distribution"] == i ].groupby('npts')['totalRuntime'].mean().reset_index()
	
	nptsDouble = np.array(dfDouble_distrib["npts"])
	nptsFloat = np.array(dfFloat_distrib["npts"])

	timeDouble = np.array(dfDouble_distrib["totalRuntime"])
	timeFloat = np.array(dfFloat_distrib["totalRuntime"])

	plt.plot(nptsDouble, timeDouble, label=dist)
	plt.plot(nptsFloat,  timeFloat, label=dist)

plt.xlabel("number of points")
plt.ylabel("time")
plt.legend()
plt.savefig("floatVsDouble.png", dpi=200)
