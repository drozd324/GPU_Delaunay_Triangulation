import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

df = pd.read_csv("./data/time.csv")

print(df)

plt.subplot(2, 2, 1)
plt.plot(df["n"], df["time"], label="my code")
plt.title("runtime complexiy")
plt.xlabel("n (number of points)")
plt.ylabel("time (seconds)")
plt.legend()

x = np.linspace(0, 1, len(df["n"]))

plt.subplot(2, 2, 2)
plt.plot(x, x**2, label=r"x^2")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, x*np.log(x), label="xlog(x)")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, x, label="x")

plt.legend()
plt.show()
