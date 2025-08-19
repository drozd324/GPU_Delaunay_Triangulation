import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

nins = np.loadtxt("./insertedPerIter.txt", dtype=int)

print(nins)
iter = np.arange(0, len(nins), 1) 

plt.plot(iter, nins)
plt.xlabel("Iteration")
plt.ylabel("Number of point insertions")

plt.savefig("ninsertVsIter.png", dpi=200)
