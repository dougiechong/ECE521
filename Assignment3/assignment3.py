import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylab import scatter, show, title, xlabel, ylabel, plot, contour, legend
import math
from utils import *

def assign_to_nearest_prob(samples, probs):
    assigns = []
    print(len(probs))
    print(len(samples))
    for i in range(len(samples)):
        hp_index = 0
        hp = probs[i][0]
        for j in range(1, len(probs[i])):
            p = probs[i][j]
            if p > hp:
                hp = p
                hp_index = j
        assigns.append(hp_index)
    return assigns

def plot(x, y, _xlabel, _ylabel):
    plt.plot(x, y)
    xlabel(_xlabel)
    ylabel(_ylabel)
    title(_ylabel + ' vs. ' + _xlabel)
    plt.tight_layout()
    plt.show()

def dist(p1, p2):
    sum = 0
    for i in range(len(p1)):
        sum += (p1[i]-p2[i])**2
    return math.sqrt(sum)

def assign_to_nearest(samples, centroids):
    assigns = []
    for i in range(len(samples)):
        min_index = 0
        min_dist = dist(samples[i], centroids[0])
        for j in range(1, len(centroids)):
            d = dist(samples[i], centroids[j])
            if d < min_dist:
                min_dist = d
                min_index = j
        assigns.append(min_index)
    return assigns

def euclideanDistance(X, Y, K, B):
    #square all elements of both matrices
    XX = tf.square(X)
    YY = tf.square(Y)
    #sum all squares of EACH sample
    XX_sum = tf.reduce_sum(XX, 1)
    YY_sum = tf.reduce_sum(YY, 1)
    #mult
    XY = 2*tf.matmul(X, Y, transpose_b=True)
    XX_reshape = tf.reshape(XX_sum, [-1,1])
    dists = tf.sub(tf.add(XX_reshape, YY_sum), XY)

    #split into K clusters, in this case 3 and find minimum
    min = tf.slice(dists, [0, 0], [B, 1])
    for i in range(1, K):
        min = tf.minimum(min, tf.slice(dists, [0, i], [B, 1]))
    return min

def logProb(X, Y, K, B, sigma, out_pi):
    #square all elements of both matrices
    XX = tf.square(X)
    YY = tf.square(Y)
    #sum all squares of EACH sample
    XX_sum = tf.reduce_sum(XX, 1)
    YY_sum = tf.reduce_sum(YY, 1)
    #mult
    XY = 2*tf.matmul(X, Y, transpose_b=True)
    XX_reshape = tf.reshape(XX_sum, [-1,1])
    print(XX_reshape)
    dists = tf.sub(tf.add(XX_reshape, YY_sum), XY)
    print(dists)
    dists = tf.reshape(dists, [B,K])

    #divide by -1/2 sigma
    dists = -tf.mul(dists,tf.inv(sigma))/2
    dists = tf.reshape(dists, [K,B])
    #reduce the log sum of the exp
    return reduce_logsumexp(tf.matmul(dists, out_pi), 0)

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
        clusters.append(plt.scatter(d[key][0], d[key][1], color=colors[key], label=percentage, alpha=0.3))
    plt.legend()

def task1(K, data, dim, showLoss, val, val_data):
    B = data.shape[0]

    dat = tf.placeholder(tf.float32)
    means = tf.Variable(tf.truncated_normal([K, dim], dtype=tf.float32))
    sigma = tf.Variable(tf.truncated_normal([B, K], dtype=tf.float32))
    out_pi = tf.Variable(tf.truncated_normal([B, K], dtype=tf.float32))
    logsoftmax(out_pi)

    #data has B samples by D=2
    loss = tf.reduce_sum(logProb(dat, means, K, B, sigma, out_pi))
    optimizer = tf.train.AdamOptimizer(0.1, 0.9, 0.99, 0.00001).minimize(loss)

    model = tf.initialize_all_variables()
    num_steps = 800
    with tf.Session() as session:
        session.run(model)
        epochs = []
        losses = []
        for step in range(num_steps):
            _, l , op= session.run([optimizer, loss, out_pi], feed_dict={dat: data})
            epochs.append(step)
            losses.append(l)
        means_values = session.run(means)
        print(means_values)
        print(op)
        #only graph if dim == 2
        if dim == 2:
            if(val):
                vdata = val_data
            else:
                vdata = data
            assigns = assign_to_nearest_prob(vdata, op)
            colors = ["red", "green", "magenta", "brown", "indigo"]
            show_clusters(assigns, vdata, colors)
            for i in range(K):
                plt.scatter(means_values[:,0][i], means_values[:,1][i], color="black", marker='x', s=50)
            title('data for ' + str(K) + ' cluster(s)')
            ylabel('data y')
            xlabel('data_x')
            plt.savefig(str(K)+'clusters.png')
            plt.show()
        if showLoss:
            print("final loss: " + str(losses[-1]))
            plot(epochs, losses, "updates", "loss")
            plt.show()

def task1_3(data):
    task1(1, data, 2, False, False, data)
    task1(2, data, 2, False, False, data)
    task1(3, data, 2, False, False, data)
    task1(4, data, 2, False, False, data)
    task1(5, data, 2, False, False, data)

def task1_4(data):
    length = math.floor(float(2*data.shape[0])/float(3))
    #train
    task1(1, data[:length], 2, True, False, data[length:])
    task1(2, data[:length], 2, True, False, data[length:])
    task1(3, data[:length], 2, True, False, data[length:])
    task1(4, data[:length], 2, True, False, data[length:])
    task1(5, data[:length], 2, True, False, data[length:])

NHIDDEN = 1
STDEV = 0.5
KMIX = 3 # number of mixtures
NOUT = KMIX * 3 # pi, mu, stdev

x = tf.placeholder(dtype=tf.float32, shape=[None,1], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[None,1], name="y")

Wh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))
bh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))

Wo = tf.Variable(tf.random_normal([NHIDDEN,NOUT], stddev=STDEV, dtype=tf.float32))
bo = tf.Variable(tf.random_normal([1,NOUT], stddev=STDEV, dtype=tf.float32))

hidden_layer = tf.nn.relu(tf.matmul(x, Wh) + bh)
output = tf.matmul(hidden_layer,Wo) + bo
#output = tf.Variable(tf.random_normal([10000,NOUT], stddev=STDEV, dtype=tf.float32))

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

def plot_error(NEPOCH, loss):
    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(100, NEPOCH,1), loss[100:], 'r-')
    plt.show()

def plot_data(x_data, y_data):
    plt.figure(figsize=(8, 8))
    plt.plot(x_data,y_data,'ro', alpha=0.3)
    plt.show()

def task2(data):
    NSAMPLE = 10000
    means = tf.Variable(tf.truncated_normal([KMIX, 2], dtype=tf.float32))

    y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
    r_data = np.float32(np.random.normal(size=(NSAMPLE,1))) # random noise
    x_data = np.float32(np.sin(0.75*y_data)*7.0+y_data*0.5+r_data*1.0)
    print(y_data)
    print(x_data)
    x_data = np.reshape(data[:,0], (len(data[:,0]), 1))
    y_data = np.reshape(data[:,1], (len(data[:,1]), 1))
    plot_data(x_data, y_data)

    out_pi, out_sigma, out_mu = get_mixture_coef(output)
    lossfunc = get_lossfunc(out_pi, out_sigma, out_mu, y)
    train_op = tf.train.AdamOptimizer().minimize(lossfunc)
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    NEPOCH = 1000
    loss = np.zeros(NEPOCH) # store the training progress here.
    for i in range(NEPOCH):
      _, outputs, out_s, out_mu_test = sess.run([train_op, out_pi, out_sigma, out_mu], feed_dict={x: x_data, y: y_data})
      print("means", out_mu_test)
      loss[i] = sess.run(lossfunc, feed_dict={x: x_data, y: y_data})
      #print(len(outputs))

    print(outputs)
    assigns = assign_to_nearest_prob(data, outputs)
    print(assigns)
    colors = ["red", "green", "magenta", "brown", "indigo"]
    show_clusters(assigns, data, colors)
    for i in range(KMIX):
        plt.scatter(out_mu_test[:,0][i], out_mu_test[:,1][i], color="black", marker='x', s=50)
        title('data for ' + str(KMIX) + ' cluster(s)')
        ylabel('data y')
        xlabel('data_x')
        plt.savefig('part2 ' + str(KMIX)+'clusters.png')
    plt.show()
    plot_error(NEPOCH, loss)
    sess.close()

def task2_2_4():
    data = np.load("data100D.npy")
    task1(3, data, 100, True, False, data)


if __name__ == "__main__":
    data = np.load("data2D.npy")
    task1(5, data, 2, True, False, data)
    #task1_3(data)
    #task1_4(data)
    #task2(data)