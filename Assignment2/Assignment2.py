import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def good_xentropy(x, t):
    return np.maximum(x, 0) - x * t + np.log(1 + np.exp(-np.abs(x)))

def task_1():
    #Placeholders
    X = tf.placeholder("float", shape=(None, 784))
    Y = tf.placeholder("float", shape=(None, 1))
    
    #Variables
    W = tf.Variable(np.random.randn(784, 1).astype("float32"), name="weight")
    b = tf.Variable(np.random.randn(1).astype("float32"), name="bias")
    
    print X.get_shape()
    
    logits = tf.add(tf.matmul(X, W), b)
    output = tf.nn.sigmoid(logits)
    
    print output.get_shape()
    
    cost_batch = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, targets=Y)
    cost = tf.reduce_mean(cost_batch)
    
    print logits.get_shape()
    print cost.get_shape()

    norm_w = tf.nn.l2_loss(W)   
    
    optimizer = tf.train.MomentumOptimizer(learning_rate=1.0, momentum=0.99)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
    train_op = optimizer.minimize(cost)
    #a hack for binary thresholding
    pred = tf.greater(output, 0.5)
    pred_float = tf.cast(pred, "float")
    
    #accuracy
    correct_prediction = tf.equal(pred_float, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print accuracy
    
    for epoch in range(2000):
        for i in xrange(8):
            x_batch = x_train[i * 100: (i + 1) * 100]
            y_batch = t_train[i * 100: (i + 1) * 100]
            cost_np, _ = sess.run([cost, train_op],
                                  feed_dict={X: x_batch, Y: y_batch})
            #Display logs per epoch step
        if epoch % 50 == 0:
            cost_train, accuracy_train = sess.run([cost, accuracy],
                                                  feed_dict={X: x_train, Y: t_train})
            cost_eval, accuracy_eval, norm_w_np = sess.run([cost, accuracy, norm_w],
                                                           feed_dict={X: x_eval, Y: t_eval})    
            print ("Epoch:%04d, cost=%0.9f, Train Accuracy=%0.4f, Eval Accuracy=%0.4f,    Norm of Weights=%0.4f" %
                   (epoch+1, cost_train, accuracy_train, accuracy_eval, norm_w_np))
    
if __name__ == "__main__":
    with np.load("notMNIST.npz") as data:
        images , labels = data["images"], data["labels"]
    
    #hello = tf.constant('Hello, TensorFlow!')
    #sess = tf.Session()
    #print sess.run(hello)
    sess = tf.InteractiveSession()
    init = tf.initialize_all_variables()
    sess.run(init)
    task_1()    