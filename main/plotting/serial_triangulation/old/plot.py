import sys
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
show_tri_labels = 0
max_pts_for_show = 10000
zoom = 0
only_save_plot = 0

if (len(sys.argv) > 1 and sys.argv[1] == "save"):
	only_save_plot = 1
	print("ONLY SAVING")
		

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

with open("tri.txt", "r") as data:
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
	print("Reading file data")
	while line_num <= line_count-2:
		read_line(data)

		iter, num_tri = np.fromstring(read_line(data), sep=" ", dtype=int) 
		iter_line_num.append(line_num)
		for i in range(num_tri):
			read_line(data)

	# interactive plot loop
	iter_idx = len(iter_line_num) - 1 if len(iter_line_num) > 100 else 1 
	run = True
	show_nbr = -1
	show_edges_to_flip = False 
	play = -1
	while run:
		goto_line(iter_line_num[iter_idx], data)
		plt.clf()
		
		print("[STORING TRANGLE INFO THIS ITERATION]")

		# collect triangle info in per iteration
		iter, num_tri = np.fromstring(read_line(data), sep=" ", dtype=int) 
		tris = np.zeros((num_tri, 10), dtype=int)
		for i in range(num_tri):
			tris[i] = np.fromstring(read_line(data), sep=" ", dtype=float) 

		print("[DRAWING TRIANGLES]")

		# plot data about each triangle
		for k, tri in enumerate(list(tris)):
			tri_pts = np.zeros((3,2))

			print(f"{k+1}/{num_tri}", end="\r")

			# plot edges of triangle
			for i in range(3):
				x0 = pts[tri[int((i  ) % 3)]][0]
				x1 = pts[tri[int((i+1) % 3)]][0]
				y0 = pts[tri[int((i  ) % 3)]][1]
				y1 = pts[tri[int((i+1) % 3)]][1]

				tri_pts[i][0] = x0
				tri_pts[i][1] = y0
					
				plt.plot([x0, x1], [y0, y1], color="black")#, linewidth=(250/num_pts))


			# scatter mean of triangles points to represent the triangle
			tri_k_avg = (np.mean(tri_pts[:, 0]), np.mean(tri_pts[:, 1]))
			#if num_pts <= max_pts_for_show:
			if show_tri_labels:
				#plt.scatter(tri_k_avg[0], tri_k_avg[1], color="green", s=1)
				plt.annotate(str(f"t{k}"), (tri_k_avg[0], tri_k_avg[1]))

			# show data about triangle k
			if show_nbr == k and iter_idx != (len(iter_line_num) - 1):

				# scatter mean of triangles points to represent the triangle
				tri_k_avg = (np.mean(tri_pts[:, 0]), np.mean(tri_pts[:, 1]))
				plt.scatter(tri_k_avg[0], tri_k_avg[1], color="green", s=100)
				plt.annotate(str(f"t{k}"), (tri_k_avg[0], tri_k_avg[1]))

				# plot lines to neighbours of triangle
				for i in range(3):
					nbr_idx = tri[3+i]
					if nbr_idx == -1:
						continue
					nbr_idx = nbr_idx % len(tris)

					# caclulate neighbours center
					tri_nbr_pts = np.zeros((3,2))			
					for j in range(3):
						pt_idx = tris[nbr_idx][j]
						tri_nbr_pts[j][0] = pts[pt_idx][0]
						tri_nbr_pts[j][1] = pts[pt_idx][1]

					tri_nbr_avg = (np.mean(tri_nbr_pts[:, 0]), np.mean(tri_nbr_pts[:, 1]))

					plt.plot([tri_k_avg[0], tri_nbr_avg[0]], [tri_k_avg[1], tri_nbr_avg[1]], color="green")

					# plotting opposite points
					opp_idx = tri[6+i]
					opp_pt = pts[tris[nbr_idx][opp_idx]]
					
					plt.plot([tri_k_avg[0], opp_pt[0]], [tri_k_avg[1], opp_pt[1]], color="blue")

			# color triangles to be flipped and fatten edge
			if show_edges_to_flip == True:
				edge_to_flip = tri[11]
				if edge_to_flip != -1:
					flip_line = [tri_pts[edge_to_flip],
								 tri_pts[(edge_to_flip+1) % 3] ]
					plt.plot([flip_line[0][0], flip_line[1][0]], [flip_line[0][1], flip_line[1][1]], linewidth=3, color="orange")
				 

		print(f"                 ", end="\r")
		print("[DRAWING POINTS]")

		# plots points in triangulation
		if num_pts <= 100 and iter_idx != (len(iter_line_num) - 1):
			if iter_idx == len(iter_line_num) - 1:
				plt.scatter(pts[:-3,0 ], pts[:-3, 1], color="red")
				for i, x, y in zip(range(num_pts), pts[:-3, 0], pts[:-3, 1]):
					plt.annotate(str(i), (x, y))
			else:
				plt.scatter(pts[:,0 ], pts[:, 1], color="red")
				for i, x, y in zip(range(num_pts), pts[:, 0], pts[:, 1]):
					plt.annotate(str(i), (x, y))

		plt.axis('square')
		if (zoom == True):
			plt.xlim(np.min([pts[i][0] for i in range(len(pts)-3)]) , np.max([pts[i][0] for i in range(len(pts)-3)]) )
			plt.ylim(np.min([pts[i][1] for i in range(len(pts)-3)]) , np.max([pts[i][1] for i in range(len(pts)-3)]) )

		plt.title(f"{num_pts} points, triangles: {num_tri}/{(len(tris))}, iter: {iter}/{len(iter_line_num) - 1}") 

		if only_save_plot:
			print("[SAVING PLOT]")
			plt.savefig("delauay_triangluation.png", dpi=max(300, int(num_pts)))
			run = False
		else:
			plt.draw()

		plt.pause(.00001)


			
		if play >= 0:
			if play == len(iter_line_num)-1:
				play = -1
			else: 
				plt.pause(4/(len(iter_line_num)-1))
				play += 1
				iter_idx += 1
			continue

		# user input
		entered = 0
		if (only_save_plot == 0):
			entered = input("> ")
		if entered == "play":
			play = 0
			iter_idx = 0

		elif get_type(entered) == int:
			if int(entered) < len(iter_line_num) and int(entered) >= -1:
				iter_idx = int(entered) % len(iter_line_num)
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
			show_edges_to_flip = 0*(show_edges_to_flip == 1) + 1*(show_edges_to_flip == 0)
		elif entered == "z":
			zoom = 0*(zoom == 1) + 1*(zoom == 0)
		elif entered == "lt":
			show_tri_labels = 0*(show_tri_labels == 1) + 1*(show_tri_labels == 0)
		elif entered == "t":
			tri_in = input(">> ")
			if get_type(tri_in) == int:
				print("idx | points        | neighbours    | opposite      | flip | insert | flipThisIter")
				print("----+---------------+---------------+---------------+------+--------+-------------")
				for i, tri in enumerate(list(tris)):
					if i == int(tri_in):
						print(f"{i:3d} | {tri[0]:3d}, {tri[1]:3d}, {tri[2]:3d} | {tri[3]:3d}, {tri[4]:3d}, {tri[5]:3d} | {tri[6]:3d}, {tri[7]:3d}, {tri[8]:3d} | {tri[9]:4d} | {tri[10]:6d} | {tri[11]:3d}")
			else:
				print("idx | points        | neighbours    | opposite      | flip | insert | flipThisIter")
				print("----+---------------+---------------+---------------+------+--------+-------------")
				for i, tri in enumerate(list(tris)):
					print(f"{i:3d} | {tri[0]:3d}, {tri[1]:3d}, {tri[2]:3d} | {tri[3]:3d}, {tri[4]:3d}, {tri[5]:3d} | {tri[6]:3d}, {tri[7]:3d}, {tri[8]:3d} | {tri[9]:4d} | {tri[10]:6d} | {tri[11]:3d}")

		elif entered == "n":
			show_nbr_in = input(">> ")
			if get_type(show_nbr_in) == int:
				show_nbr = int(show_nbr_in)
			elif "n":
				show_nbr = (int(show_nbr) + 1) % (len(tris))
			else:
				continue
			

		elif entered == "q" or entered == "quit" or entered == "^C":
			run = False
		elif entered == "h" or entered == "help":
			print("======[User Guide]======") 
			print("| f: move forward one iteration") 
			print("| b: move backward one iteration") 
			print(f"| int in [0, {len(iter_line_num)-1}]: go to iteration with given number") 
			print("| -1: goes to the end of the seqence") 
			print("| t: prints traingles plotted") 
			print("| n: displays neighbours of chosen triangle") 
			print("| lt: toggles labels on triangles")
			print("| e: shows edges marked for flipping") 
			print("| play: loops through all iterations") 
			print("| z: zoom") 
			print("| q: quit this session") 
			print("| h: prints this user guide") 
		else:
			continue
				
	plt.ioff()

