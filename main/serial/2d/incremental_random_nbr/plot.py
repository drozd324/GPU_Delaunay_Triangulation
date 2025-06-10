import numpy as np
import matplotlib.pyplot as plt

from ast import literal_eval
def get_type(input_data):
    try:
        return type(literal_eval(input_data))
    except (ValueError, SyntaxError):
        return str

line_num = 0
line_count = 0

print("")
print("==========[INITIALISING PLOT]==========")

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

	plt.ion()
	plt.show(block=False)

	iter_line_num = []
	while line_num <= line_count-2:
		read_line(data)
		plt.clf()

		iter, num_tri = np.fromstring(read_line(data), sep=" ", dtype=int) 
		iter_line_num.append(line_num)
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
		#plt.xlim(0, 1)
		#plt.ylim(0, 1)

		plt.title(f"iter: {iter}") 
		plt.draw()
		plt.pause(.00001)
		plt.savefig(f"./data/plots/plot_{iter}")
		print(f"% done: {round(100 * (line_num/line_count), 2)}", end='\r')

	print("                                               ", end='\r')

	iter_idx = len(iter_line_num) -1
	run = True
	while run:
		goto_line(iter_line_num[iter_idx], data)
		plt.clf()

		iter, num_tri = np.fromstring(read_line(data), sep=" ", dtype=int) 
		tris = np.zeros((num_tri, 3), dtype=int)
		for i in range(num_tri):
			tris[i] = np.fromstring(read_line(data), sep=" ", dtype=float) 

		print("TRIANGLES PLOTTED")
		for tri in list(tris):
			for i in range(3):
				x0 = pts[tri[int((i  ) % 3)]][0]
				x1 = pts[tri[int((i+1) % 3)]][0]
				y0 = pts[tri[int((i  ) % 3)]][1]
				y1 = pts[tri[int((i+1) % 3)]][1]
					
				plt.plot([x0, x1], [y0, y1], color="black")

			print(tri[0], tri[1], tri[2])

		plt.scatter(pts[:,0 ], pts[:, 1], color="red")
		if num_pts < 20:
			for i, x, y in zip(range(num_pts), pts[:, 0], pts[:, 1]):
				plt.annotate(str(i), (x, y))

		plt.axis('square')
		#plt.xlim(0, 1)
		#plt.ylim(0, 1)

		plt.title(f"iter: {iter}") 
		plt.draw()
		plt.pause(.00001)

		entered = input("> ")
		if get_type(entered) == int:
			if int(entered) < len(iter_line_num) and int(entered) >= 0:
				iter_idx = int(entered)
			else:
				print(f"INVALID INDEX: must be in [0, {len(iter_line_num)-1}]")

		elif entered == "+":
			if iter_idx == len(iter_line_num)-1:
				print(f"INVALID INDEX: must be in [0, {len(iter_line_num)-1}]")
			else:
				iter_idx += 1
		elif entered == "-":
			if iter_idx == 0:
				print(f"INVALID INDEX: must be in [0, {len(iter_line_num)-1}]")
			else:
				iter_idx -= 1

		elif entered == "e":
			iter_idx = len(iter_line_num)-1

		elif entered == "q" or entered == "quit" or entered == "^C":
			run = False
		elif entered == "h" or entered == "help":
			print("======[User Guide]======") 
			print("| +: move forward one iteration") 
			print("| -: move backward one iteration") 
			print(f"| int in [0, {len(iter_line_num)-1}]: go to iteration with given number") 
			print("| e: goes to the end of the seqence") 
			print("| q: quit this session") 
			print("| h: prints this user guide") 
				
	#print(f"% done: 100", end='\r')
	plt.ioff()
