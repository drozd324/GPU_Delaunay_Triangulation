import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv("./data/time.csv")

avg_df = df.groupby('n')['time'].mean().reset_index()
print(avg_df)

plt.subplot(3, 1, 1)
plt.loglog(avg_df["n"], avg_df["time"], label="my code")
plt.loglog( [list(avg_df["n"])[0], list(avg_df["n"])[-1]], [list(avg_df["time"])[0], list(avg_df["time"])[-1]], label="f(x)=x")
plt.title("loglog plot")
plt.xlabel("n (number of points)")
plt.ylabel("time (seconds)")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(avg_df["n"], avg_df["time"], label="my code")
plt.plot( [list(avg_df["n"])[0], list(avg_df["n"])[-1]], [list(avg_df["time"])[0], list(avg_df["time"])[-1]], label="f(x)=x")
plt.title("basic plot")
plt.xlabel("n (number of points)")
plt.ylabel("time (seconds)")
plt.legend()

plt.subplot(3, 1, 3)
x = np.linspace(list(avg_df["n"])[0], list(avg_df["n"])[-1], 100)
plt.loglog(x, np.log(x), label="xlog(x)")
plt.legend()

#plt.subplot(3, 1, 2)
#plt.loglog(x, x**2, label=r"x^2")
#plt.legend()

#plt.show()

