import numpy as np
import matplotlib.pyplot as plt

pts = np.loadtxt("circ.txt")
num_pts = len(pts)

for iter in range(num_pts-3):

	plt.scatter(pts[:,0 ], pts[:, 1], color="red")
	for i, x, y in zip(range(num_pts), pts[:, 0], pts[:, 1]):
		plt.annotate(str(i), (x, y))

	plt.plot(np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100)))

	plt.show();
