import numpy as np
import matplotlib.pyplot as plt

pts = np.loadtxt("./data/points.txt")
num_pts = len(pts)

for iter in range(num_pts):
	plt.clf()

	plt.scatter(pts[:,0 ], pts[:, 1], color="red")
	for i, x, y in zip(range(num_pts), pts[:, 0], pts[:, 1]):
		plt.annotate(str(i), (x, y))

	triangles = np.loadtxt(f"./data/triangles_iterations/triangles_{iter}.txt", dtype=int)

	for tri in triangles:
		for i in range(3):
			x0 = pts[tri[int((i  ) % 3)]][0]
			x1 = pts[tri[int((i+1) % 3)]][0]
			y0 = pts[tri[int((i  ) % 3)]][1]
			y1 = pts[tri[int((i+1) % 3)]][1]
				
			plt.plot([x0, x1], [y0, y1], color="black")

	plt.savefig(f"./data/triangles_iterations_plots/plot_{iter}", dpi=200)

