import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X = tf.placeholder("float", [None, 784])
Y_ = tf.placeholder("float", [None, 10])
keep_prob = tf.placeholder("float")

def add_layer(inputs, in_shape, out_shape):
    W = tf.Variable(tf.random_normal([in_shape, out_shape],stddev=0.01))
    b = tf.Variable(tf.constant(0.12,shape=[1,out_shape]))
    Y = tf.matmul(inputs, W) + b
    return Y

x_train, y_train, = mnist.train.images, mnist.train.labels
hidden_layer1 = tf.nn.relu(add_layer(X,784,10))
hidden_drop = tf.nn.dropout(hidden_layer1, keep_prob)
predictions = add_layer(hidden_layer1, 10, 10)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions,
                       labels=Y_))
train = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
predict_step = tf.argmax(predictions, 1)

x_test, y_test, = mnist.test.images, mnist.test.labels
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        for j, k in zip(range(0, len(x_train), 101), range(101, len(y_train)+1, 101)):
            sess.run(train, {X: x_train[j:k], Y_: y_train[j:k], keep_prob:0.8})
        print(i, np.mean(np.argmax(y_test, axis=1) ==
                         sess.run(predict_step, ={X: x_test,keep_prob:1.0})))
