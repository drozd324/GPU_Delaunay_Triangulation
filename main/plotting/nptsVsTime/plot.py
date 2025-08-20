import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv("./data.csv")

avg_df = df.groupby('seed')['totalRuntime'].mean().reset_index()

dist_names = ["Uniform", "Clustered center", "Clustered boudary", "Gaussian"]
for i, dist in enumerate(dist_names):
	df_distrib = df[ df["distribution"] == i ].groupby('npts')['totalRuntime'].mean().reset_index()
	
	npts = list(df_distrib["npts"])
	time = list(df_distrib["totalRuntime"])
	
	plt.loglog(npts, time, label=dist)

plt.xlabel("Number of points")
plt.ylabel("Time (s)")
plt.legend()

plt.savefig("nptsVsTime.png", dpi=200)
