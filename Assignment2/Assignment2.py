# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from IPython.display import display, Image
from scipy import ndimage
#from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import matplotlib.cm as cm
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import math

#for debugging purposes
def print_letter(image): 
    plt.figure(1)
    plt.imshow(image, cmap = cm.Greys_r)
    plt.grid(True)
    plt.show()    

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def stochastic_grad_descent(train_images, val_images, test_images, num_labels, val_labels, image_size):
    '''main code used from http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/2_fullyconnected.ipynb'''
    batch_size = 128
    
    graph = tf.Graph()
    with graph.as_default():
    
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(val_images)
        tf_test_dataset = tf.constant(test_images)
    
        # Variables.
        weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
        # Optimizer.
        optimizer = tf.train.MomentumOptimizer(0.1, 0.2).minimize(loss)
    
        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
        
    num_steps = 3001
    
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_images[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), val_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

# Model for 1 layer RELU
def relu_model(_X, _weights, _biases):
    #Hidden layer with RELU activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) 
    return tf.matmul(layer_1, _weights['out']) + _biases['out']

# Model for 2 layer RELU
def relu_model_two_layers(_X, _weights, _biases):
    #Hidden layer with RELU activation
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) 
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) 
    return tf.matmul(layer_2, _weights['out']) + _biases['out']

def task2(num_h_units, model_type):
    '''main code used from http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/2_fullyconnected.ipynb'''
    batch_size = 128
    log_likelihood = []
    epochs = []
    
    graph = tf.Graph()
    with graph.as_default():
    
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(val_images)
        tf_test_dataset = tf.constant(test_images)
    
        depth = 10
        # Variables.
        
        # Network Parameters
        n_hidden_1 = num_h_units # 1st layer num features
        n_hidden_2 = num_h_units # 2nd layer num features
        n_input = 784 # MNIST data input (img shape: 28*28)
        n_classes = 10 # MNIST total classes (0-9 digits)        
        # Store layers weight & bias
        
        if(model_type == 'relu'):
            weights = {
                'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
                'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'out': tf.Variable(tf.random_normal([n_classes]))
            }
            model = relu_model
        elif(model_type == 'relu_two_layers'):
            weights = {
                'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
                'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
                'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
            }
            biases = {
                'b1': tf.Variable(tf.random_normal([n_hidden_1])),
                'b2': tf.Variable(tf.random_normal([n_hidden_2])),
                'out': tf.Variable(tf.random_normal([n_classes]))
            }       
            model = relu_model_two_layers
        # Training computation.
        logits = model(tf_train_dataset, weights, biases)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
        # Optimizer.
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
    
        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset, weights, biases))
        test_prediction = tf.nn.softmax(model(tf_test_dataset, weights, biases))
        
    num_steps = 1001
    
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_images[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            epochs.append(step)
            log_likelihood.append(-l)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), val_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
    plt.plot(epochs, log_likelihood)
    xlabel('epochs')
    ylabel('log_likelihood')
    plt.show()

def task3():
    #call task2 with different number of hidden units
    for h_unit in [100, 500, 1000]:
        task2(train_images, val_images, test_images, num_labels, val_labels, image_size, h_unit, 'relu')

def task4():
    #call task2 with different model function
    task2(500, 'relu_two_layers')

if __name__ == "__main__":
    with np.load("notMNIST.npz") as data:
        images , labels = data["images"], data["labels"]
    
    reformatted = np.zeros((18720, 28,28))
    for i in range(len(reformatted)):
        reformatted[i] = images[:,:,i]
    
    train_images = reformatted[:15000]
    val_images = reformatted[15000:16000]
    test_images = reformatted[16000:]
    labels = labels.flatten()
    train_labels = labels[:15000]
    val_labels = labels[15000:16000]
    test_labels = labels[16000:]
    
    print('Training set', train_images.shape, train_labels.shape)
    print('Validation set', val_images.shape, val_labels.shape)
    print('Test set', test_images.shape, test_labels.shape)
    
    image_size = 28
    num_labels = 10
    num_channels = 1 # grayscale
    
    train_images, train_labels = reformat(train_images, train_labels)
    val_images, val_labels = reformat(val_images, val_labels)
    test_images, test_labels = reformat(test_images, test_labels)
    
    print('Training set', train_images.shape, train_labels.shape)
    print('Validation set', val_images.shape, val_labels.shape)
    print('Test set', test_images.shape, test_labels.shape)
    # With gradient descent training, even this much data is prohibitive.
    # Subset the training data for faster turnaround.
    train_subset = 15000
    
    #stochastic_grad_descent(train_images, val_images, test_images, num_labels, val_labels, image_size)
    #task2(1000, 'relu')
    #task3()
    #task4()