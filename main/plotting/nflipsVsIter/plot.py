import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
from ast import literal_eval

line_num = 0
def read_line(data):
	global line_num
	line_num += 1
	return data.readline()

def goto_line(num, data):
	data.seek(0);
	for i in range(num-1):
		data.readline()

nflips_initer = []
iter = 0
sum_nflips_initer = []
timings_nflips_initer  = []

plt.figure(figsize=(10, 4))
with open("flipedPerIter.txt", "r") as data:
	total_iter = len(data.readlines()) // 3
	data.seek(0)

	for i in range(total_iter):
		arr1 = np.fromstring(read_line(data), sep=" ", dtype=int) 
		arr2 = np.fromstring(read_line(data), sep=" ", dtype=float) 
		read_line(data)
		
		nflips_initer.extend(arr1)
		sum_nflips_initer.append(sum(arr1))
		#timings_nflips_initer.extend(arr2)


ax1 = plt.subplot(1, 2, 1)
plt.title("Total number of flips in an iteration")
plt.bar(np.arange(0, len(sum_nflips_initer), 1, dtype=int), sum_nflips_initer,
		color="cornflowerblue", edgecolor="black", hatch="//");
plt.xlabel("Total number of flips")
plt.ylabel("Number of flips")
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

ax2 = plt.subplot(1, 2, 2)
plt.title("Number of flips in each pass")
plt.bar(np.arange(0, len(nflips_initer), 1, dtype=int), nflips_initer,
		color="cornflowerblue", edgecolor="black", hatch="//");
plt.xlabel("Pass of paralell flipping")
#plt.ylabel("Number of flips")
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))


plt.savefig("nflipsVsIter.png", dpi=200)



