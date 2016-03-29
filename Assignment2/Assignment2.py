# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib
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

def task7():
    # x and y are placeholders for our training data
    x = tf.placeholder("float")
    y = tf.placeholder("float")
    # w is the variable storing our values. It is initialised with starting "guesses"
    # w[0] is the "a" in our equation, w[1] is the "b"
    w = tf.Variable([1.0, 2.0], name="w")
    # Our model of y = a*x + b
    y_model = tf.mul(x, w[0]) + w[1]

    # Our error is defined as the square of the differences
    error = tf.square(y - y_model)
    # The Gradient Descent Optimizer does the heavy lifting
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

    # Normal TensorFlow - initialize values, create a session and run the model
    model = tf.initialize_all_variables()

    with tf.Session() as session:
        session.run(model)
        for i in range(1000):
            x_value = np.random.rand()
            y_value = x_value * 2 + 6
            session.run(train_op, feed_dict={x: x_value, y: y_value})

        w_value = session.run(w)
        print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))


NHIDDEN = 24
STDEV = 0.5
KMIX = 2 # number of mixtures
NOUT = KMIX * 3 # pi, mu, stdev

x = tf.placeholder(dtype=tf.float32, shape=[None,1], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[None,1], name="y")

Wh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))
bh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))

Wo = tf.Variable(tf.random_normal([NHIDDEN,NOUT], stddev=STDEV, dtype=tf.float32))
bo = tf.Variable(tf.random_normal([1,NOUT], stddev=STDEV, dtype=tf.float32))

hidden_layer = tf.nn.tanh(tf.matmul(x, Wh) + bh)
output = tf.matmul(hidden_layer,Wo) + bo

def get_mixture_coef(output):
  out_pi = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_sigma = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_mu = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")

  out_pi, out_sigma, out_mu = tf.split(1, 3, output)

  max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
  out_pi = tf.sub(out_pi, max_pi)

  out_pi = tf.exp(out_pi)

  normalize_pi = tf.inv(tf.reduce_sum(out_pi, 1, keep_dims=True))
  out_pi = tf.mul(normalize_pi, out_pi)

  out_sigma = tf.exp(out_sigma)

  return out_pi, out_sigma, out_mu

out_pi, out_sigma, out_mu = get_mixture_coef(output)

NSAMPLE = 2500

y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
r_data = np.float32(np.random.normal(size=(NSAMPLE,1))) # random noise
x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)

plt.figure(figsize=(8, 8))
plt.plot(x_data,y_data,'ro', alpha=0.3)
plt.show()

x_test = np.float32(np.arange(-15,15,0.1))
NTEST = x_test.size
x_test = x_test.reshape(NTEST,1) # needs to be a matrix, not a vector

def show_clusters(assigns, vdata, colors):
    d = {}
    for i, num in enumerate(assigns):
        if(d.has_key(num)):
            d[num][0].append(vdata[i][0])
            d[num][1].append(vdata[i][1])
        else:
            d[num] = [[vdata[i][0]], [vdata[i][1]]]
    clusters = []
    for key in d:
        percentage = str(100*float(len(d[key][0]))/float(len(vdata))) + "%"
        clusters.append(plt.scatter(d[key][0], d[key][1], color=colors[key], label=percentage))
    plt.legend()

def assign_to_nearest_prob(samples, probs):
    assigns = []
    print(len(probs))
    print(len(samples))
    for i in range(len(probs)):
        hp_index = 0
        hp = probs[i][0]
        for j in range(1, len(probs[i])):
            p = probs[i][j]
            if p > hp:
                hp = p
                hp_index = j
        assigns.append(hp_index)
    return assigns

oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi) # normalisation factor for gaussian, not needed.
def tf_normal(y, mu, sigma):
  result = tf.sub(y, mu)
  result = tf.mul(result,tf.inv(sigma))
  result = -tf.square(result)/2
  return tf.mul(tf.exp(result),tf.inv(sigma))*oneDivSqrtTwoPI

def get_lossfunc(out_pi, out_sigma, out_mu, y):
  result = tf_normal(y, out_mu, out_sigma)
  result = tf.mul(result, out_pi)
  result = tf.reduce_sum(result, 1, keep_dims=True)
  result = -tf.log(result)
  return tf.reduce_mean(result)

def get_pi_idx(x, pdf):
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if accumulate >= x:
      return i
  print('error with sampling ensemble')
  return -1

def generate_ensemble(out_pi, out_mu, out_sigma, M = 10):
  NTEST = x_test.size
  result = np.random.rand(NTEST, M) # initially random [0, 1]
  rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
  mu = 0
  std = 0
  idx = 0

  # transforms result into random ensembles
  for j in range(0, M):
    for i in range(0, NTEST):
      idx = get_pi_idx(result[i, j], out_pi[i])
      mu = out_mu[i, idx]
      std = out_sigma[i, idx]
      result[i, j] = mu + rn[i, j]*std
  return result

lossfunc = get_lossfunc(out_pi, out_sigma, out_mu, y)
train_op = tf.train.AdamOptimizer().minimize(lossfunc)

def task8():
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    NEPOCH = 10000
    loss = np.zeros(NEPOCH) # store the training progress here.
    for i in range(NEPOCH):
      _, out_pi_test, out_sigma_test, out_mu_test = sess.run([train_op, out_pi, out_sigma, out_mu],feed_dict={x: x_data, y: y_data})
      loss[i] = sess.run(lossfunc, feed_dict={x: x_data, y: y_data})

    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(100, NEPOCH,1), loss[100:], 'r-')
    plt.show()

    #out_pi_test2, out_sigma_test2, out_mu_test2 = sess.run(get_mixture_coef(output), feed_dict={x: x_test})

    dat = []
    for i in range(len(x_data)):
        dat.append([x_data[i][0], y_data[i][0]])
    print(len(out_pi_test))
    print(dat)
    assigns = assign_to_nearest_prob(dat, out_pi_test)
    print(assigns)
    colors = []
    for name, hex in matplotlib.colors.cnames.iteritems():
        colors.append(name)
    print(colors)
    show_clusters(assigns, dat, colors)
    plt.show()

    '''y_test = generate_ensemble(out_pi_test2, out_mu_test2, out_sigma_test2)

    plt.figure(figsize=(8, 8))
    plt.plot(x_data,y_data,'ro', x_test,y_test,'bo',alpha=0.3)
    plt.show()'''


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
    #task7()
    task8()