import numpy as np
import matplotlib.pyplot as plt

x = []
y = []

r = 1; 
n = 10000

for _ in range(0, n):
#	theta = np.random.uniform() * np.pi
#	phi	  = np.random.uniform() *2*np.pi
	
#	x.append(r*np.sin(theta)*np.cos(phi))
#	y.append(r*np.sin(theta)*np.sin(phi))

	r = np.random.uniform()
	#r = r**(1/4)
	r = np.sqrt(np.sqrt(r))
	theta = np.random.uniform() * 2 * np.pi

	x.append(r*np.cos(theta))
	y.append(r*np.sin(theta))

plt.scatter(x, y, s=0.5)
plt.axis("square")
plt.show()

