import numpy as np
import matplotlib.pyplot as plt

nins = np.loadtxt("./insertedPerIter.txt", dtype=int)
iter = np.arange(len(nins))

ratios = [nins[i+1] / nins[i] for i in range(len(nins)-1)]

#fig, ax1 = plt.subplots(figsize=(8, 5))
fig, ax1 = plt.subplots()

bars = ax1.bar(iter, nins, color="cornflowerblue", edgecolor="black", hatch="//")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Number of point insertions")
ax1.tick_params(axis="y")

ax2 = ax1.twinx()
line, = ax2.plot(iter[1:], ratios, color="red", marker="o", label="ratio")
ax2.set_ylabel("Insertion ratio")
ax2.tick_params(axis="y")

handles = [bars, line]
labels = ["Number of insertions", "Insertion ratio"]
ax1.legend(handles, labels, loc="center left")

plt.tight_layout()
plt.savefig("ninsertVsIter.png", dpi=200)

