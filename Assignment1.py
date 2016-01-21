#!/usr/bin/python
import numpy as np
import math
import operator

from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import matplotlib.pyplot as plt


# most KNN functions obtained from here 
#http://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
#euclidian distance function
#use numpy.broadcasts
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

#get neighbours function
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-2
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
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
 
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def getValidationErrors(testSet, predictions):
	validationErrors = 0
	for x in range(len(testSet)):
		if testSet[x][-1] != predictions[x]:
			validationErrors += 1
	return validationErrors

train_x = np.linspace(1.0, 10.0, num=100)[:, np.newaxis]
train_y = np.sin(train_x) + 0.1*np.power(train_x, 2) + 0.5*np.random.randn(100, 1)

with np.load("TINY_MNIST.npz") as data:
	x, t = data["x"], data["t"]
	x_eval, t_eval = data["x_eval"], data["t_eval"]

k = 1
n_values = [5, 50, 100, 200, 400, 800]
validation_errors = []
table = [[5, 0], [50, 0], [100, 0], [200, 0], [400, 0], [800, 0]]

print x.size/64
print t.size
print t_eval.size

x_full = []
x_eval_full = []
for i in range(len(x)):
	x_full.append(np.append(x[i],t[i]))

for i in range(len(x_eval)):
	x_eval_full.append(np.append(x_eval[i],t_eval[i]))

def diff_n(x_full, x_eval_full, k, validation_errors):
	for n in n_values:
		trainingSet=x_full[0:n]
		testSet=x_eval_full
		predictions=[]
		for i in range(len(testSet)):
			neighbors = getNeighbors(trainingSet, testSet[i], k)
			result = getResponse(neighbors)
			predictions.append(result)
		validation_errors.append(getValidationErrors(testSet,predictions))

#diff_n(x_full, x_eval_full, k, validation_errors)

#Larger N generally results in less validation errors
print "N   Validation Errors"
for i in range(len(validation_errors)):
	print str(n_values[i]) + "   " + str(validation_errors[i])

k_values = [1, 3, 5, 7, 21, 101, 401]
validation_errors = []
def diff_k(x_full, x_eval_full, validation_errors):
	for k in k_values:
		trainingSet=x_full
		testSet=x_eval_full
		predictions=[]
		for i in range(len(testSet)):
			neighbors = getNeighbors(trainingSet, testSet[i], k)
			result = getResponse(neighbors)
			predictions.append(result)
		print k
		validation_errors.append(getValidationErrors(testSet,predictions))
#diff_k(x_full, x_eval_full, validation_errors)

#smaller k performed better
print "K   Validation Errors"
for i in range(len(validation_errors)):
	print str(k_values[i]) + "   " + str(validation_errors[i])

a = []
b = []
for i in range(len(train_x)):
	a.append(train_x[i][0])
for i in range(len(train_y)):
	b.append(train_y[i][0])

#have to use gradient descent
'''coeffs = np.polyfit(a, b, 1)
coeffs_5 = np.polyfit(a, b, 5)
ffit = np.poly1d(coeffs)
ffit_5 = np.poly1d(coeffs_5)
x_new = np.linspace(a[0], a[-1], num=len(a)*10)
scatter(train_x, train_y, marker='o', c='b')
title('training set')
xlabel('train_x')
ylabel('train_y')
plt.plot(x_new, ffit(x_new))
plt.plot(x_new, ffit_5(x_new))
plt.show()
show()'''


def sigmaSum(x, w, n, b, t):
	total = 0
	for i in range(n):
		total += math.pow((np.dot(w, x[i]) + b - t[i][0]), 2)
	return total

learning_rate = 0.1 #try values 0.0001-0.1
n = 200
b = np.random.rand()
w = np.random.rand(64) #initialize weights randomly
#print np.dot(w, x[2])
euclidean_cost = (1.0/(2.0*n))*sigmaSum(x, w, n, b, t)
#print euclidean_cost
threshold = 0.5

def lin_regress(w, b, x, targ, x_eval, t_eval, T, epochs, lmbda):					
	for i in range(epochs): #arbitrary number of epochs
		for k in range(T):
			for j in range(len(w)):
				total_j = 0;
				total_j += (targ[k][0]-np.dot(w,x[k])-b)*x[k][j]
				total_j -= w[j]*lmbda
				w[j] += learning_rate*total_j
	validation_errors = 0
	max_t = 0
	min_t = 0
	for x_ind in x_eval:
		if(np.dot(w, x_ind)+b > max_t):
			max_t = np.dot(w, x_ind)+b
		if(np.dot(w, x_ind)+b < min_t):
			min_t = np.dot(w, x_ind)+b
	threshold = 0.5
	for i in range(len(x_eval)):
		if((np.dot(w, x_eval[i])+b)> threshold):
			target = 1.0
		else:
			target = 0.0
		if(t_eval[i][0] != target):
			validation_errors += 1
			
	print "error cost", (1.0/(2.0*T))*sigmaSum(x, w, n, b, targ)
	return validation_errors

for T in [100, 200, 400, 800]:
	validation_errors = lin_regress(w, b, x, t, x_eval, t_eval, T, 100, 0)
	print T, ' ', validation_errors

'''for epoch in range(50):
	validation_errors = lin_regress(w, b, x, t, x_eval, t_eval, 50, epoch, 0)
	training_errors = lin_regress(w, b, x, t, x[0:49], t[0:49], 50, epoch, 0)
	print 'epoch', epoch, ' validation errors: ', validation_errors
	print 'epoch', epoch, ' training errors: ', training_errors'''

'''for lmbda in [0, 0.0001, 0.001, 0.01, 0.1, 0.5]:
	validation_errors = lin_regress(w, b, x, t, x_eval, t_eval, 50, 100, lmbda)
	print 'lmbda', lmbda, ' validation errors: ', validation_errors'''


#reccommends 10 000 epochs

#find gradient, adjust w and b until euclidean cost is minimized


