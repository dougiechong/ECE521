#!/usr/bin/python
import numpy as np
import math
import operator

from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import matplotlib.pyplot as plt


#euclidian distance function
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

def fit(X, Y):

    def mean(Xs):
        return sum(Xs) / len(Xs)
    m_X = mean(X)
    m_Y = mean(Y)

    def std(Xs, m):
        normalizer = len(Xs) - 1
        return math.sqrt(sum((pow(x - m, 2) for x in Xs)) / normalizer)
    # assert np.round(Series(X).std(), 6) == np.round(std(X, m_X), 6)

    def pearson_r(Xs, Ys):

        sum_xy = 0
        sum_sq_v_x = 0
        sum_sq_v_y = 0

        for (x, y) in zip(Xs, Ys):
            var_x = x - m_X
            var_y = y - m_Y
            sum_xy += var_x * var_y
            sum_sq_v_x += pow(var_x, 2)
            sum_sq_v_y += pow(var_y, 2)
        return sum_xy / math.sqrt(sum_sq_v_x * sum_sq_v_y)
    # assert np.round(Series(X).corr(Series(Y)), 6) == np.round(pearson_r(X, Y), 6)

    r = pearson_r(X, Y)

    b = r * (std(Y, m_Y) / std(X, m_X))
    A = m_Y - b * m_X

    def line(x):
        return b * x + A
    return line

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

a = np.array([1])
b = np.array([1])
for i in range(len(train_x)):
	np.append(a, train_x[i])
for i in range(len(train_y)):
	np.append(b, train_y[i])
print a
t = np.arange(0.0, 1.2, 0.1)
fit_line = np.polyfit(a, b, 1)
scatter(train_x, train_y, marker='o', c='b')
title('training set')
xlabel('train_x')
ylabel('train_y')
plt.plot(fit_line)
plt.show()
show()
print t


