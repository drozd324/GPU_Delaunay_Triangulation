import numpy as np
import matplotlib.pyplot as plt

line_num = 0
line_count = 0

def read_line(data):
	global line_num
	line_num += 1
	return data.readline()

def goto_line(num, data):
	data.seek(0);
	for i in range(num-1):
		data.readline()

#def plot_iter(data)	

with open("./data/data.txt", "r") as data:
	line_count = len(data.readlines())
	data.seek(0)

	num_pts = int(read_line(data))
	pts = np.zeros((num_pts, 2), dtype=float)
	for i in range(num_pts):
		pts[i] = np.fromstring(read_line(data), sep=" ", dtype=float) 


	plt.scatter(pts[:,0 ], pts[:, 1], color="red")
	if num_pts < 20:
		for i, x, y in zip(range(num_pts), pts[:, 0], pts[:, 1]):
			plt.annotate(str(i), (x, y))

	plt.axis('square')
	#plt.xlim(0, 1)
	#plt.ylim(0, 1)

	plt.title(f"iter: {iter}") 
	plt.show()
