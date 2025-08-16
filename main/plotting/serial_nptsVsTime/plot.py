import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv("./data.csv")

avg_df = df.groupby('seed')['totalRuntime'].mean().reset_index()

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
dist_names = ["uniform", "clustered center", "clustered boundary", "gaussian"]
for i, dist in enumerate(dist_names):
	df_distrib = df[ df["distribution"] == i ].groupby('npts')['totalRuntime'].mean().reset_index()
	
	npts = list(df_distrib["npts"])
	time = list(df_distrib["totalRuntime"])

	#plt.plot(npts, time, label=dist)
	plt.loglog(npts, time, label=dist)

plt.xlabel("number of points")
plt.ylabel("time (seconds)")
plt.legend()
plt.savefig("serial_nptsVsTime.png", dpi=200)
#plt.show()
