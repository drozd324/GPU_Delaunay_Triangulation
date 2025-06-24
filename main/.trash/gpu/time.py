import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv("./data/time.csv")

plt.subplot(3, 1, 1)
plt.loglog(df["n"], df["time"], label="my code")
plt.title("runtime complexiy")
plt.xlabel("n (number of points)")
plt.ylabel("time (seconds)")
plt.legend()

x = np.linspace(1, 10, 100)

plt.subplot(3, 1, 2)
plt.loglog(x, x**2, label=r"x^2")
plt.legend()

plt.subplot(3, 1, 3)
plt.loglog(x, np.log(x), label="xlog(x)")
plt.legend()

plt.show()
