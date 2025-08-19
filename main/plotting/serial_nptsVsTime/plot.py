import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv("./data.csv")

avg_df = df.groupby('seed')['totalRuntime'].mean().reset_index()

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
dist_names = ["uniform", "clustered center", "clustered boundary", "gaussian"]
coeffs = 0, 1, 2
for i, dist in enumerate(dist_names):
	df_distrib = df[ df["distribution"] == i ].groupby('npts')['totalRuntime'].mean().reset_index()
	
	npts = np.array(list(df_distrib["npts"]))
	time = np.array(list(df_distrib["totalRuntime"]))

	#x^2 fit
	coeffs = np.polyfit(npts, time, 2)

	plt.plot(npts, time, label=dist, linestyle="-")

	#plt.loglog(npts, time, label=dist)
	#plt.loglog(npts, np.polyval(coeffs, npts), label="square fit")

plt.plot(npts, np.polyval(coeffs, npts), color="black", label="square fit", linestyle=":")
plt.xlabel("number of points")
plt.ylabel("time (seconds)")
plt.legend()
plt.savefig("serial_nptsVsTime.png", dpi=200)
