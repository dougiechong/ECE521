# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.cm as cm
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
import math
import random

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

def get_errors(predictions, labels):
    return np.sum(np.argmax(predictions, 1) != np.argmax(labels, 1))

# Model 
def model(_X, _weights, _biases, _dropout):
    #Hidden layer with RELU activation
    #initalize with data
    layers = [_X]
    for layer in range(len(_weights)-1):
        layers.append(tf.nn.relu(tf.add(tf.matmul(layers[layer], _weights[layer]), _biases[layer]))) 
    if(_dropout):
        return tf.matmul(tf.nn.dropout(layers[-1], 0.5), _weights['out']) + _biases['out']
    else:
        return tf.matmul(layers[-1], _weights['out']) + _biases['out']

def task1(num_h_units, learning_rate, num_layers, dropout):
    '''main code used from http://nbviewer.jupyter.org/github/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/2_fullyconnected.ipynb'''
    batch_size = 128
    log_likelihood = []
    val_log_likelihood = []
    train_errors = []
    val_errors = []
    epochs = []
    min_cost = 1000000
    
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
        n_hidden_3 = num_h_units # 2nd layer num features
        n_input = 784 # MNIST data input (img shape: 28*28)
        n_classes = 10 # MNIST total classes (0-9 digits)        
        # Store layers weight & bias
        weights = {
            'out' : tf.Variable(tf.random_normal([n_input, n_classes]))
        }
        biases = {
            0 : tf.Variable(tf.random_normal([n_hidden_1])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        
        if(num_layers == 1):
            weights[0] = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
            weights['out'] = tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
        if(num_layers == 2):
            weights[0] = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
            weights[1] = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
            weights['out'] = tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
            biases[1] = tf.Variable(tf.random_normal([n_hidden_2]))
        if(num_layers == 3):
            weights[0] = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
            weights[1] = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
            weights[2] = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3]))
            weights['out'] = tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
            biases[1] = tf.Variable(tf.random_normal([n_hidden_2]))
            biases[2] = tf.Variable(tf.random_normal([n_hidden_3]))
        
        # Training computation.
        logits = model(tf_train_dataset, weights, biases, dropout)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        val_logits = model(tf_valid_dataset, weights, biases, dropout)
        val_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(val_logits, val_labels))
        
        # Optimizer.
        if(num_layers == 0):
            optimizer = tf.train.MomentumOptimizer(0.1, 0.2).minimize(loss)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset, weights, biases, False))
        test_prediction = tf.nn.softmax(model(tf_test_dataset, weights, biases, False))
        
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
            _, l, vl, predictions = session.run([optimizer, loss, val_loss, train_prediction], feed_dict=feed_dict)
            epochs.append(step)
            log_likelihood.append(-l)
            val_log_likelihood.append(-vl)
            train_errors.append(get_errors(predictions, batch_labels))
            val_error = get_errors(valid_prediction.eval(), val_labels)
            val_errors.append(val_error)
            
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            if (min_cost < val_error):
                best_epoch = step
                if(step>20):
                    break
            min_cost = val_error
        print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), val_labels))        
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
        test_errors = get_errors(test_prediction.eval(), test_labels)
        final_val_errors = get_errors(valid_prediction.eval(), val_labels)
        print("Test errors:  %d" %  test_errors)
        print("Validation errors:  %d" %  final_val_errors)
        print("Best Epoch:  %d" %  best_epoch)
    plot(epochs, log_likelihood, 'epoch', 'training log likelihood')
    plot(epochs, val_log_likelihood, 'epoch', 'validation log likelihood')
    plot(epochs, train_errors, 'epoch', 'training errors')
    plot(epochs, val_errors, 'epoch', 'validation errors')
    
def plot(x, y, _xlabel, _ylabel):
    plt.plot(x, y)
    xlabel(_xlabel)
    ylabel(_ylabel)
    title(_ylabel + ' vs. ' + _xlabel)
    plt.tight_layout()
    plt.show()
    
def task2():
    #call task1 with 1 hidden layer
    task1(1000, 0.01, 1, False)
    
def task3():
    #call task1 with different number of hidden units
    for h_unit in [100, 500, 1000]:
        task1(h_unit, 0.01, 0, False)
    for h_unit in [100, 500, 1000]:
        task1(h_unit, 0.01, 1, False)

def task4():
    #call task1 with two layers
    task1(500, 0.01, 2, False)
    
def task5():
    #call task1 with dropout
    task1(500, 0.0005, 2, True)
    
def task6():
    #call 5 times
    for i in range(5):
        #randomize some parameters
        num_layers = random.randint(1,3)
        hidden_units_per_layer = random.randint(100,500)
        if(round(random.random()) == 1.0):
            dropout = True
        else:
            dropout = False
        learning_rate = math.e**(random.uniform(-4,-2))
        print(num_layers)
        print(hidden_units_per_layer)
        print(learning_rate)
        print(dropout)
        #call task1 with randomized parameters
        task1(hidden_units_per_layer, learning_rate, num_layers, dropout)

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
    
    image_size = 28
    num_labels = 10
    
    train_images, train_labels = reformat(train_images, train_labels)
    val_images, val_labels = reformat(val_images, val_labels)
    test_images, test_labels = reformat(test_images, test_labels)
    
    # Subset the training data for faster turnaround.
    train_subset = 15000
    
    #task1(1000, 0.01, 0, False)
    #task2()
    #task3()
    #task4()
    #task5()
    #task6()