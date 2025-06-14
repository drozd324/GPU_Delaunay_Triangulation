import numpy as np
import matplotlib.pyplot as plt

line_num = 0
line_count = 0
with open("./data/data.txt") as data:
	line_count = len(data.readlines())

with open("./data/data.txt") as data:
	line_num = 0
	
	num_pts = int(data.readline())
	line_num += 1
	pts = np.zeros((num_pts, 2), dtype=float)
	for i in range(num_pts):
		pts[i] = np.fromstring(data.readline(), sep=" ", dtype=float) 
		line_num += 1

	while line_num <= line_count-2:
		data.readline()
		line_num += 1
		plt.clf()

		plt.scatter(pts[:,0 ], pts[:, 1], color="red")
		for i, x, y in zip(range(num_pts), pts[:, 0], pts[:, 1]):
			plt.annotate(str(i), (x, y))

		iter, num_tri = np.fromstring(data.readline(), sep=" ", dtype=int) 
		tris = np.zeros((num_tri, 3), dtype=int)
		line_num += 1
		for i in range(num_tri):
			tris[i] = np.fromstring(data.readline(), sep=" ", dtype=float) 
			line_num += 1

		for tri in list(tris):
			for i in range(3):
				x0 = pts[tri[int((i  ) % 3)]][0]
				x1 = pts[tri[int((i+1) % 3)]][0]
				y0 = pts[tri[int((i  ) % 3)]][1]
				y1 = pts[tri[int((i+1) % 3)]][1]
					
				plt.plot([x0, x1], [y0, y1], color="black")

		plt.axis('square')
		plt.xlim(0, 1)
		plt.ylim(0, 1)
	
		plt.title(f"incircle check {iter}") 
		plt.savefig(f"./data/plots/plot_{iter}", dpi=200)
		print(f"Line number: {line_num} out of {line_count}", end='\r')

	print(f"Line number: {line_count} out of {line_count}")
