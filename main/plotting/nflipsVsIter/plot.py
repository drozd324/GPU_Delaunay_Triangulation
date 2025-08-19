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

with open("flipedPerIter.txt", "r") as data:
	total_iter = len(data.readlines()) // 3
	data.seek(0)

	for i in range(total_iter):
		arr1 = np.fromstring(read_line(data), sep=" ", dtype=int) 
		arr2 = np.fromstring(read_line(data), sep=" ", dtype=int) 
		read_line(data)
		
		nflips_initer.extend(arr1)
		sum_nflips_initer.append(arr2[1])

plt.subplot(1, 2, 1)
plt.title("Sum of flips in an iteration")
plt.plot(np.arange(0, len(sum_nflips_initer), 1), sum_nflips_initer);

plt.subplot(1, 2, 1)
plt.title("Sum of flips in an iteration")
plt.plot(np.arange(0, len(nflips_initer), 1), nflips_initer)

plt.savefig("flipedPerIter.png", dpi=200)
