from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


style.use('ggplot')

#updating value of m
def dcdm_calc(xs, y_predict, y_orig):
	return sum(2*(y_predict - y_orig)*xs)

#updating b
def dcdb_calc(y_predict, y_orig):
	return sum(2*(y_predict - y_orig))

def training_fn(xs, y_orig, m, b):
	cost = 0
	y_predict = []

	#predicting values
	for x in xs:
		y = m*x + b
		y_predict.append(y)

	#getting cost for prediction	
	cost  = sum((y_predict - y_orig)**2)
	#print(cost)

	#back propagation
	dcdm = dcdm_calc(xs, y_predict, y_orig)
	dcdb = dcdb_calc(y_predict, y_orig)

	m = m - 0.001*dcdm
	b = b - 0.001*dcdb
	return m,b

def run():
	#dataset
	xs = np.array([6.2,6.5,5.48,6.54,7.93], dtype=np.float64)
	y_orig = np.array([26.3,26.65,25.03,26.01,30.47], dtype=np.float64)

	#initial values for m and b
	m = 0 #gradient
	b = 0 #y intecept

	#best fit line using linear regression
	regr_line = []

	#training loop
	for i in range(100000):
		m,b = training_fn(xs, y_orig, m, b)

	#building regression line using m and b values
	for x in xs:
		y = x*m + b
		regr_line.append(y)

	#output and testing a point
	print ('m={}, b={}'.format(m,b))
	y_test = m*7.18 + b
	print('the diff is {}'.format(27.9 - y_test))
	plt.scatter(xs, y_orig)
	plt.plot(xs, regr_line)
	plt.plot(7.18, y_test, 'g+') 
	plt.plot(7.18, 27.9, 'b+') 
	plt.show()

if __name__ == '__main__':
	run()