import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv("./data.csv")

avg_df = df.groupby('ntpb')['totalRuntime'].mean().reset_index()

dist_names = ["uniform square", "uniform disk", "gaussian"]
for i, dist in enumerate(dist_names):
	df_distrib = df[ df["distribution"] == i ].groupby('npts')['totalRuntime'].mean().reset_index()
	
	npts = list(df_distrib.iloc[:, 0])
	time = list(df_distrib.iloc[:, 1])

	plt.plot(npts, time, label=dist)

plt.xlabel("number of points")
plt.ylabel("time (seconds)")
plt.legend()
plt.savefig("ntbpVstime.png", dpi=200)
plt.show()
