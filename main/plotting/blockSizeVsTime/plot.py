import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv("./data.csv")

avg_df = df.groupby('ntpb')['totalRuntime'].mean().reset_index()

dist_names = ["uniform", "clustered center", "clustered boudary", "gaussian"]
for i, dist in enumerate(dist_names):
	df_distrib = df[ df["distribution"] == i ].groupby('ntpb')['totalRuntime'].mean().reset_index()
	
	ntpb = list(df_distrib["ntpb"])
	time = list(df_distrib["totalRuntime"])

	plt.plot(ntpb, time, label=dist)

plt.xlabel("number of threads per block")
plt.ylabel("time (seconds)")
plt.legend()
plt.savefig("blockSizeVsTime.png", dpi=200)
#plt.show()
