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
    XY = 2*tf.matmul(X, Y, transpose_b=True)
    XX_reshape = tf.reshape(XX_sum, [tf.shape(XX_sum).eval()[0],1])
    dists = tf.sub(tf.add(XX_reshape, YY_sum), XY)

    #print(XX.eval())
    #print(XX_sum.eval())
    #print(XX_reshape.eval())
    #print(YY_sum.eval())
    #print(XY.eval())
    #print(dists.eval())
    return dists

def nump_compute_dist(a, b):
    #square all elements of both matrices
    a_squared = np.square(a)
    b_squared = np.square(b)
    #sum all squares of EACH sample
    a_sum_square = np.sum(a_squared,axis=1)
    b_sum_square = np.sum(b_squared,axis=1)
    #multiply each sample by each cluster
    mul = np.dot(a,b.T)
    dists = a_sum_square[:,np.newaxis]+b_sum_square-2*mul
    return dists

def make_test_vectors():
    l = []
    for i in range(5):
        l.append([])
        for j in range(3):
            l[i].append(i*5+j)
    C = tf.constant(l)
    l = []
    for i in range(10):
        l.append([])
        for j in range(3):
            l[i].append(i*5+j)
    D = tf.constant(l)
    return [C, D]

def task1():
    data = np.load("data2D.npy")
    graph = tf.Graph()
    with graph.as_default():
        A = tf.random_normal([5, 3])
        B = tf.random_normal([3, 10])
        l = []
        for i in range(5):
            l.append([])
            for j in range(3):
                l[i].append(i*5+j)
        C = tf.constant(l)
        l = []
        for i in range(10):
            l.append([])
            for j in range(3):
                l[i].append(i*5+j)
        D = tf.constant(l)

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        x = A.eval()
        print(C.eval())
        print(tf.transpose(D).eval())
        #takes in 2 matrices BxD and KxD

        print(euclideanDistance(C,D).eval())
        a = np.array([[1,1,1,1],[2,2,2,2]])
        b = np.array([[1,2,3,4],[1,1,1,1],[1,2,1,9]])

        #square all elements of both matrices
        a_squared = np.square(a)
        b_squared = np.square(b)
        #sum all squares of EACH sample
        a_sum_square = np.sum(a_squared,axis=1)
        b_sum_square = np.sum(b_squared,axis=1)
        #multiply each sample by each cluster
        mul = np.dot(a,b.T)
        print(a_sum_square)
        print(a_sum_square[:,np.newaxis])
        dists = a_sum_square[:,np.newaxis]+b_sum_square-2*mul

        #print(nump_compute_dist(a,b))

if __name__ == "__main__":
    data = np.load("data2D.npy")
    #for i in range(len(data)):
     #   print(data[i])
    print(data)
    task1()