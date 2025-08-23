import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from matplotlib.ticker import MultipleLocator

df = pd.read_csv("./data.csv")

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
dist_names = ["Uniform", "Clustered center", "Clustered boundary", "Gaussian"]
for i, dist in enumerate(dist_names):
	df_distrib = df[ df["distribution"] == i ].groupby('ntpb')['totalRuntime'].mean().reset_index()
	
	ntpb = list(df_distrib["ntpb"])
	time = list(df_distrib["totalRuntime"])

	plt.plot(ntpb, time, label=dist)

plt.gca().xaxis.set_major_locator(MultipleLocator(64))


plt.xlabel("Number of threads per block")
plt.ylabel("Time (s)")
plt.legend()
plt.savefig("blockSizeVsTime.png", dpi=200)
#plt.show()
