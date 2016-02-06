'''
Go to bottom of code and uncomment each task inidvidually to run
'''
import numpy as np
import math
import operator

from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import matplotlib.pyplot as plt


# most KNN functions obtained from here 
#http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

#euclidian distance function
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

#get neighbours function
def getNeighbors(trainingSet, testInstance, k):
	dists = []
	length = len(testInstance)-2
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		dists.append((trainingSet[x], dist))
	dists.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(dists[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def getValidationErrors(testSet, predictions):
	validationErrors = 0
	for x in range(len(testSet)):
		if testSet[x][-1] != predictions[x]:
			validationErrors += 1
	return validationErrors

def stochasticGradientDescent(x, y, theta, alpha, m, numIterations, lmbda, val=False):
	m = len(x)
	#go through number of iterations
	for i in range(numIterations):
		#go though each sample
		for j in range(len(y)):
			hyp = np.dot(x, theta)
			loss = hyp - y		
			#go through each dimension of sample
			gradient = np.dot(x[j], loss[j])
			gradient += lmbda*theta
			cost = np.sum(loss ** 2) / (2 * m) + (lmbda/2.0) * np.dot(theta, theta)
			#print("Iteration %d | Cost: %f" % (i*len(loss)+j, cost))
			theta -= alpha * gradient
		if((val == True) and (i%100==0)):
			validationErrors = validateSet(x_eval, t_eval, theta)
			if(i==0):
				scatter(i, validationErrors, marker='o', c='b', label='validation errors')
			else:
				scatter(i, validationErrors, marker='o', c='b')
			trainingErrors = validateSet(addOnes(x[:49,1:]), t[:49], theta, True)
			if(i==0):
				scatter(i, trainingErrors, marker='o', c='r', label='training errors')
			else:
				scatter(i, trainingErrors, marker='o', c='r')
	return theta

def plot(x,y):
	m, n = np.shape(x)
	theta = np.random.rand(n) #initialize weights randomly
	theta = stochasticGradientDescent(x, y, theta, alpha, m, numIterations, 0)
	line_y_values = np.dot(theta, x.transpose())
	scatter(train_x, train_y, marker='o', c='b')
	title('training set')
	xlabel('train_x')
	ylabel('train_y')
	plt.plot(train_x, line_y_values)
	plt.show()
	show()	
	
train_x = np.linspace(1.0, 10.0, num=100)[:, np.newaxis]
train_y = np.sin(train_x) + 0.1*np.power(train_x, 2) + 0.5*np.random.randn(100, 1)

with np.load("TINY_MNIST.npz") as data:
	x, t = data["x"], data["t"]
	x_eval, t_eval = data["x_eval"], data["t_eval"]

#setup parameters
k = 1
n_values = [5, 50, 100, 200, 400, 800]
table = [[5, 0], [50, 0], [100, 0], [200, 0], [400, 0], [800, 0]]
x_full = []
x_eval_full = []
for i in range(len(x)):
	x_full.append(np.append(x[i],t[i]))
for i in range(len(x_eval)):
	x_eval_full.append(np.append(x_eval[i],t_eval[i]))
	
def task1():
	validation_errors = []
	for n in n_values:
		trainingSet=x_full[0:n]
		testSet=x_eval_full
		predictions=[]
		for i in range(len(testSet)):
			neighbors = getNeighbors(trainingSet, testSet[i], k)
			result = getResponse(neighbors)
			predictions.append(result)
		validation_errors.append(getValidationErrors(testSet,predictions))
	#Larger N generally results in less validation errors
	print "N   Validation Errors"
	for i in range(len(validation_errors)):
		print str(n_values[i]) + "   " + str(validation_errors[i])

def task2():
	k_values = [1, 3, 5, 7, 21, 101, 401]
	validation_errors = []
	for k in k_values:
		trainingSet=x_full
		testSet=x_eval_full
		predictions=[]
		for i in range(len(testSet)):
			neighbors = getNeighbors(trainingSet, testSet[i], k)
			result = getResponse(neighbors)
			predictions.append(result)
		validation_errors.append(getValidationErrors(testSet,predictions))
	#smaller k performed better
	print "K   Validation Errors"
	for i in range(len(validation_errors)):
		print str(k_values[i]) + "   " + str(validation_errors[i])

#function to add a column of ones to an array to use dot product and w0
def addOnes(a):
	b = np.zeros((np.shape(a)[0],np.shape(a)[1]+1))
	b[:,0] = 1.0
	b[:,1:] = a
	return b

def validateSet(x_eval, t_eval, w, training = False):
	validation_errors = 0
	threshold = 0.5
	if(training == False):
		x_eval = addOnes(x_eval)
	for i in range(len(x_eval)):
		if((np.dot(w, x_eval[i]))> threshold):
			target = 1.0
		else:
			target = 0.0
		#print target, t_eval[i][0], np.dot(w, x_eval[i])
		if(t_eval[i][0] != target):
			validation_errors += 1
			
	return validation_errors	
	

def linRegress(x, targ, x_eval, t_eval, T, epochs, lmbda, val=False):
	#add ones column to front
	x_new = addOnes(x)
	m, n = np.shape(x_new) 
	w = np.random.rand(n)
	#perform stochastic gradient descent
	w = stochasticGradientDescent(x_new[0:T], targ.flatten()[0:T], w, 0.001, m, epochs, lmbda, val)
	'''validation_errors = 0
	threshold = 0.5
	x_eval = addOnes(x_eval)
	for i in range(len(x_eval)):
		if((np.dot(w, x_eval[i]))> threshold):
			target = 1.0
		else:
			target = 0.0
		#print target, t_eval[i][0], np.dot(w, x_eval[i])
		if(t_eval[i][0] != target):
			validation_errors += 1'''
	return validateSet(x_eval, t_eval, w)

def genData(numPoints, bias, variance, train_x, train_y):
	x = np.zeros(shape=(numPoints, 2))
	y = np.zeros(shape=numPoints)
	# basically a straight line
	for i in range(0, numPoints):
		# bias feature
		x[i][0] = 1
		x[i][1] = train_x[i][0]
		# our target variable
		y[i] = train_y[i][0]
	return x, y

def genDataFive(numPoints, bias, variance, train_x, train_y):
	x = np.zeros(shape=(numPoints, 6))
	y = np.zeros(shape=numPoints)
	# basically a straight line
	for i in range(6):
		if(i==0):
			std = 1
		else:
			std = (train_x**i).std()
		x[:,i] = ((train_x**i)/std).flatten()
	for i in range(0, numPoints):		
		# our target variable
		y[i] = train_y[i][0]
	return x, y

numIterations= 10000
alpha = 0.0005
	
def task3():
	x, y = genData(100, 25, 10, train_x, train_y)
	plot(x,y);

def task4():
	x, y = genDataFive(100, 25, 10, train_x, train_y)
	plot(x,y);
	
def task5():
	threshold = 0.5
	for T in [100, 200, 400, 800]:
		validation_errors = linRegress(x, t, x_eval, t_eval, T, 1000, 0)
		print T, ' ', validation_errors

def task6():
	validation_errors = linRegress(x, t, x_eval, t_eval, 50, 10000, 0, True)
	title('errors vs epoch')
	xlabel('epoch')
	ylabel('errors')
	plt.legend(loc='upper right')
	plt.show()	

def task7():
	validation_errors = []
	lmbdas = [0, 0.0001, 0.001, 0.01, 0.1, 0.5]
	for lmbda in lmbdas:
		validation_errors.append(linRegress(x, t, x_eval, t_eval, 50, 10000, lmbda))
	scatter(lmbdas, validation_errors, marker='o', c='b')
	print validation_errors
	title('validation errors vs lambda')
	xlabel('lambda')
	ylabel('validation errors')
	plt.show()

#task1()
#task2()
#task3()
task4()
#task5()
#task6()
#task7()