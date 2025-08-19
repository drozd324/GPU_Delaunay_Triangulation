import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv("./data.csv")

avg_df = df.groupby('seed')['totalRuntime'].mean().reset_index()

plt.figure(figsize=(10, 4))
dist_names = ["Uniform", "Clustered center", "Clustered boudary", "Gaussian"]
for i, dist in enumerate(dist_names):
	df_distrib = df[ df["distribution"] == i ].groupby('npts')['totalRuntime'].mean().reset_index()
	
	npts = list(df_distrib["npts"])
	time = list(df_distrib["totalRuntime"])
	
	plt.subplot(1,2,1)
	plt.plot(npts, time, label=dist)

	plt.subplot(1,2,2)
	#plt.loglog(npts, time, label=dist)
	plt.semilogx(npts, time, label=dist)

plt.subplot(1,2,1)
plt.xlabel("Number of points")
plt.ylabel("Time (s)")
plt.legend()

plt.subplot(1,2,2)
plt.xlabel("Number of points")
#plt.ylabel("Time (s)")
plt.legend()

plt.savefig("nptsVsTime.png", dpi=200)
