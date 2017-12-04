import matplotlib.pyplot as plt
import numpy as np
import pylab

data = [[3,   1.5, 1],
		[2,   1,   0],
		[4,   1.5, 1],
		[3,   1,   0], 
		[3.5, 0.5, 1],
		[2,   0.5, 0],
		[5.5, 1,   1],
		[1,   1,   0]]

mystery_flower = [4.5, 1]

w1 = np.random.randn() #weight of first input
w2 = np.random.randn() #weight of second input
b = np.random.randn() #bias

print(w1, w2, b)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_p(x):
	return sigmoid(x) * (1-sigmoid(x))

T = np.linspace(-6, 6, 100)
plt.plot(T, sigmoid(T), c = 'r')
plt.plot(T, sigmoid_p(T), c = 'b')

#scatter plot of data
for i in range(len(data)):
	point = data[i]
	color = "r"
	if point[2] == 0:
		color = "b"
	plt.scatter(point[0], point[1], c = color)
#training loop

for i in range(1, 1000):
	ri = np.random.randint(len(data))
	point = data[ri]	

plt.show()