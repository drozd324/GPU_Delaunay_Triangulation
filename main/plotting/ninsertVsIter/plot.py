import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

nins = np.loadtxt("./insertedPerIter.txt", dtype=int)

iter = np.arange(0, len(nins), 1) 

plt.bar(iter, nins,
		color="cornflowerblue", edgecolor="black", hatch="//");
plt.xlabel("Iteration")
plt.ylabel("Number of point insertions")

plt.savefig("ninsertVsIter.png", dpi=200)
