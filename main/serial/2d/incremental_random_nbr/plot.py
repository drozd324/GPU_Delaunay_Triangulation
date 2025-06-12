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

with open("./data/data.txt", "r") as data:
	line_count = len(data.readlines())
	data.seek(0)

	# collect points of triangulation
	num_pts = int(read_line(data))
	pts = np.zeros((num_pts, 2), dtype=float)
	for i in range(num_pts):
		pts[i] = np.fromstring(read_line(data), sep=" ", dtype=float) 

	# activate interavtive plot
	plt.ion()
	plt.show(block=False)

	# collect file layout data
	iter_line_num = []
	while line_num <= line_count-2:
		read_line(data)

		iter, num_tri = np.fromstring(read_line(data), sep=" ", dtype=int) 
		iter_line_num.append(line_num)
		for i in range(num_tri):
			read_line(data)

	# interactive plot loop
	iter_idx = 1# len(iter_line_num) - 1
	run = True
	show_nbr = -1
	while run:
		goto_line(iter_line_num[iter_idx], data)
		plt.clf()

		# collect triangle info in per iteration
		iter, num_tri = np.fromstring(read_line(data), sep=" ", dtype=int) 
		tris = np.zeros((num_tri, 9), dtype=int)
		for i in range(num_tri):
			tris[i] = np.fromstring(read_line(data), sep=" ", dtype=float) 

		# plot data about each triangle
		for k, tri in enumerate(list(tris)):
			tri_pts = np.zeros((3,2))

			# plot edges of triangle
			for i in range(3):
				x0 = pts[tri[int((i  ) % 3)]][0]
				x1 = pts[tri[int((i+1) % 3)]][0]
				y0 = pts[tri[int((i  ) % 3)]][1]
				y1 = pts[tri[int((i+1) % 3)]][1]

				tri_pts[i][0] = x0
				tri_pts[i][1] = y0
					
				plt.plot([x0, x1], [y0, y1], color="black")


			# scatter mean of triangles points to represent the triangle
			tri_k_avg = (np.mean(tri_pts[:, 0]), np.mean(tri_pts[:, 1]))
			plt.scatter(tri_k_avg[0], tri_k_avg[1], color="green", s=1)
			plt.annotate(str(f"t{k}"), (tri_k_avg[0], tri_k_avg[1]))

			if (show_nbr == k):
				# scatter mean of triangles points to represent the triangle
				tri_k_avg = (np.mean(tri_pts[:, 0]), np.mean(tri_pts[:, 1]))
				plt.scatter(tri_k_avg[0], tri_k_avg[1], color="green", s=100)
				plt.annotate(str(f"t{k}"), (tri_k_avg[0], tri_k_avg[1]))

				# plot lines to neighbours of triangle
				for i in range(3):
					nbr_idx = tri[3+i]
					if nbr_idx == -1:
						continue

					# caclulate neighbours center
					tri_nbr_pts = np.zeros((3,2))			
					for j in range(3):
						pt_idx = tris[nbr_idx][j]
						tri_nbr_pts[j][0] = pts[pt_idx][0]
						tri_nbr_pts[j][1] = pts[pt_idx][1]

					tri_nbr_avg = (np.mean(tri_nbr_pts[:, 0]), np.mean(tri_nbr_pts[:, 1]))

					plt.plot([tri_k_avg[0], tri_nbr_avg[0]], [tri_k_avg[1], tri_nbr_avg[1]], color="green")
					

		plt.scatter(pts[:,0 ], pts[:, 1], color="red")
		if num_pts < 20:
			for i, x, y in zip(range(num_pts), pts[:, 0], pts[:, 1]):
				plt.annotate(str(i), (x, y))

		plt.axis('square')
		#plt.xlim(0, 1)
		#plt.ylim(0, 1)

		plt.title(f"{num_pts} points, iter: {iter}") 
		plt.draw()
		plt.pause(.00001)

		entered = input("> ")
		if get_type(entered) == int:
			if int(entered) < len(iter_line_num) and int(entered) >= 0:
				iter_idx = int(entered)
			else:
				print(f"INVALID INDEX: must be in [0, {len(iter_line_num)-1}]")

		elif entered == "f":
			if iter_idx == len(iter_line_num)-1:
				print(f"INVALID INDEX: must be in [0, {len(iter_line_num)-1}]")
			else:
				iter_idx += 1
		elif entered == "b":
			if iter_idx == 0:
				print(f"INVALID INDEX: must be in [0, {len(iter_line_num)-1}]")
			else:
				iter_idx -= 1

		elif entered == "e":
			iter_idx = len(iter_line_num)-1
					
		elif entered == "t":
			for i, tri in enumerate(list(tris)):
				print(i, "|", tri[0], tri[1], tri[2], " ", tri[3], tri[4], tri[5] , " ", tri[6], tri[7], tri[8])

		elif entered == "n":
			show_nbr = int(input("tri: "))

		elif entered == "q" or entered == "quit" or entered == "^C":
			run = False
		elif entered == "h" or entered == "help":
			print("======[User Guide]======") 
			print("| f: move forward one iteration") 
			print("| b: move backward one iteration") 
			print(f"| int in [0, {len(iter_line_num)-1}]: go to iteration with given number") 
			print("| e: goes to the end of the seqence") 
			print("| t: prints traingles plotted") 
			print("| n: prints neighbours of chosen triangle") 
			print("| q: quit this session") 
			print("| h: prints this user guide") 
		else:
			continue
				
	#print(f"% done: 100", end='\r')
	plt.ioff()
