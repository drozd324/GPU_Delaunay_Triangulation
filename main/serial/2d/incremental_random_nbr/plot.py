import numpy as np
import matplotlib.pyplot as plt

line_num = 0
line_count = 0

def read_line(data):
	global line_num
	line_num += 1
	return data.readline()

with open("./data/data.txt") as data:
	line_count = len(data.readlines())

data = open("./data/data.txt")
num_pts = int(read_line(data))
pts = np.zeros((num_pts, 2), dtype=float)
for i in range(num_pts):
	pts[i] = np.fromstring(read_line(data), sep=" ", dtype=float) 

plt.ion()
plt.show(block=False)
while line_num <= line_count-2:
	read_line(data)
	plt.clf()

	iter, num_tri = np.fromstring(read_line(data), sep=" ", dtype=int) 
	tris = np.zeros((num_tri, 3), dtype=int)
	for i in range(num_tri):
		tris[i] = np.fromstring(read_line(data), sep=" ", dtype=float) 

	for tri in list(tris):
		for i in range(3):
			x0 = pts[tri[int((i  ) % 3)]][0]
			x1 = pts[tri[int((i+1) % 3)]][0]
			y0 = pts[tri[int((i  ) % 3)]][1]
			y1 = pts[tri[int((i+1) % 3)]][1]
				
			plt.plot([x0, x1], [y0, y1], color="black")

	plt.scatter(pts[:,0 ], pts[:, 1], color="red")
	if num_pts < 20:
		for i, x, y in zip(range(num_pts), pts[:, 0], pts[:, 1]):
			plt.annotate(str(i), (x, y))

	plt.axis('square')
	plt.xlim(0, 1)
	plt.ylim(0, 1)

	plt.title(f"incircle check {iter}") 
	plt.draw()
	plt.pause(.00001)
	#plt.savefig(f"./data/plots/plot_{iter}")
	print(f"% done: {round(100 * (line_num/line_count), 2)}", end='\r')

print(f"% done: 100", end='\r')
plt.ioff()
plt.show()
plt.savefig(f"./data/plots/plot.png")
