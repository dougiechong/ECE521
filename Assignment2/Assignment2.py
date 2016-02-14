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

def print_letter(image): 
    plt.figure(1)
    plt.imshow(image, cmap = cm.Greys_r)
    plt.grid(True)
    
    plt.show()    

with np.load("notMNIST.npz") as data:
    images , labels = data["images"], data["labels"]

reformatted = np.zeros((18720, 28,28))
for i in range(len(reformatted)):
    reformatted[i] = images[:,:,i]

for i in range(len(reformatted)):
    #print(reformatted[i])
    #print_letter(reformatted[i])
    #print(labels[i])
    break
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

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_images, train_labels = reformat(train_images, train_labels)
val_images, val_labels = reformat(val_images, val_labels)
test_images, test_labels = reformat(test_images, test_labels)

print('Training set', train_images.shape, train_labels.shape)
print('Validation set', val_images.shape, val_labels.shape)
print('Test set', test_images.shape, test_labels.shape)
# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 15000

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def grad_descent(train_images, train_labels, val_images, test_images, train_subset, image_size, num_labels):
    graph = tf.Graph()
    with graph.as_default():
    
        # Input data.
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        tf_train_dataset = tf.constant(train_images[:train_subset, :])
        tf_train_labels = tf.constant(train_labels[:train_subset])
        tf_valid_dataset = tf.constant(val_images)
        tf_test_dataset = tf.constant(test_images)
    
        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random valued following a (truncated)
        # normal distribution. The biases get initialized to zero.
        weights = tf.Variable(
            tf.truncated_normal([image_size * image_size, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))
    
        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
        # Optimizer.
        # We are going to find the minimum of this loss using gradient descent.
        optimizer = tf.train.MomentumOptimizer(0.1, 0.2).minimize(loss)
    
        # Predictions for the training, validation, and test data.
        # These are not part of training, but merely here so that we can report
        # accuracy figures as we train.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(
            tf.matmul(tf_valid_dataset, weights) + biases)
        test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
        
    num_steps = 801
    
    with tf.Session(graph=graph) as session:
        # This is a one-time operation which ensures the parameters get initialized as
        # we described in the graph: random weights for the matrix, zeros for the
        # biases. 
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            # Run the computations. We tell .run() that we want to run the optimizer,
            # and get the loss value and the training predictions returned as numpy
            # arrays.
            _, l, predictions = session.run([optimizer, loss, train_prediction])
            if (step % 100 == 0):
                print('Loss at step %d: %f' % (step, l))
                print('Training accuracy: %.1f%%' % accuracy(
                    predictions, train_labels[:train_subset, :]))
                # Calling .eval() on valid_prediction is basically like calling run(), but
                # just to get that one numpy array. Note that it recomputes all its graph
                # dependencies.
                print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(), val_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

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

if __name__ == "__main__":
    #grad_descent(train_images, train_labels, val_images, test_images, train_subset, image_size, num_labels)
    stochastic_grad_descent(train_images, val_images, test_images, num_labels, val_labels, image_size)