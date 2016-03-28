import tensorflow as tf
import numpy as np

def euclideanDistance(X, Y):
    #square all elements of both matrices
    XX = tf.square(X)
    YY = tf.square(Y)
    #sum all squares of EACH sample
    XX_sum = tf.reduce_sum(XX, 1)
    YY_sum = tf.reduce_sum(YY, 1)
    #mult
    #print(X.eval())
    #print(Y.eval())
    XY = 2*tf.matmul(X, Y, transpose_b=True)
    XX_reshape = tf.reshape(XX_sum, [-1,1])
    dists = tf.sub(tf.add(XX_reshape, YY_sum), XY)

    #print(XX.eval())
    #print(XX_sum.eval())
    #print(XX_reshape.eval())
    #print(YY_sum.eval())
    #print(XY.eval())
    #print(dists.eval())
    #just to get shape

    #split into K clusters, in this case 3 and find minimum
    split0, split1, split2 = tf.split(1, 3, dists)
    mins = tf.minimum(split0, split1)
    mins = tf.minimum(mins, split2)
    return mins

def task1():
    data = np.load("data2D.npy")
    graph = tf.Graph()
    with graph.as_default():
        A = tf.random_normal([5, 3])
        B = tf.random_normal([3, 10])
        l = []
        for i in range(5):
            l.append([])
            for j in range(2):
                to_add = np.float64(i*5+j)
                l[i].append(to_add)
        C = tf.constant(l)
        l = []
        for i in range(10):
            l.append([])
            for j in range(2):
                to_add = np.float64(i*5+j)
                l[i].append(to_add)
        D = tf.constant(l)



        #randomly set k means
        #means = tf.Variable(tf.truncated_normal([3, 2], dtype=tf.float32))

        #set 0s to compare euclidean distances against
        labels = tf.Variable(tf.zeros([10000, 1], dtype=tf.float32))
        means = tf.placeholder(tf.float32, shape=(3, 2))

        #data has B samples by D=2
        dat = tf.constant(data, dtype=tf.float32)
        logits = euclideanDistance(dat, means)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, labels))
        mean_prediction = tf.nn.softmax(logits)
        #optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        optimizer = tf.train.AdamOptimizer(0.1, 0.9, 0.99, 0.00001).minimize(loss)

    num_steps = 3
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        all_means = tf.Variable(tf.truncated_normal([3, 2], dtype=tf.float32))
        #for step in range(num_steps):
        rslt = euclideanDistance(dat,means)
        print(rslt.eval())
        for step in range(num_steps):
            print(means.eval())
            feed_dict = {means: all_means}
            _, l, predictions = session.run([optimizer, loss, mean_prediction], feed_dict=feed_dict)
            #takes in 2 matrices BxD and KxD



if __name__ == "__main__":
    data = np.load("data2D.npy")
    print(data)
    task1()