# Fizz Buzz in Tensorflow! 
# see http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/ 

import numpy as np
import tensorflow as tf

# fizz_buzz_encode
num_digits = 11
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

def fizz_buzz_encode(i):
    if i % 15 == 0: return np.array([0, 0, 0, 1]) # FizzBuzz
    elif i % 5 == 0: return np.array([0, 0, 1, 0]) # Buzz
    elif i % 3 == 0: return np.array([0, 1, 0, 0]) # Fizz
    else: return np.array([1, 0, 0, 0]) # number
# train data
X_train = np.array([binary_encode(i,num_digits) for i in range(101,2**num_digits)])
Y_train = np.array([fizz_buzz_encode(i) for i in range(101, 2**num_digits)])
# test data
X_test = np.array([binary_encode(i,num_digits) for i in range(1, 101)])
Y_test = np.array([fizz_buzz_encode(i) for i in range(1, 101)])
# input layer & out layer
x = tf.placeholder(tf.float64, shape=[None,num_digits])
y = tf.placeholder(tf.float64, shape=[None,4])
keep_prob = tf.placeholder(tf.float64)
# define layer function
def add_layer(inputs, isize, osize):
    W = tf.Variable(tf.random_normal(shape=[isize,osize], stddev=0.01,
                    dtype=tf.float64))
    Y = tf.matmul(inputs,W)
    return Y
# neural network
hidden_layer1 = tf.nn.relu(add_layer(x,num_digits,512))
hidden_layer2 = tf.nn.relu(add_layer(hidden_layer1,512,256))
hidden_drop = tf.nn.dropout(hidden_layer2, keep_prob)
# prediction y
prediction = add_layer(hidden_drop,256,4)
# test function
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_prediction = sess.run(prediction, {x:v_xs, keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_prediction,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))
    # result  = sess.run(accuracy, {x_: v_xs, y_:v_ys keep_prob:1})
    result = accuracy.eval({x: v_xs, y:v_ys, keep_prob:1})
    return result

# train !!!!!
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                               logits=prediction, labels=y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
# initial
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# start
for i in range(3000):
    # feed train data
    for j in range(0, len(X_train), 128):
        end = j + 128
        sess.run(train_step, feed_dict={x:X_train[j:end], y:Y_train[j:end],
        keep_prob:0.8})
    if i %100==0:
        print(i,'train:',compute_accuracy(X_train,Y_train),
        'test:',compute_accuracy(X_test,Y_test))
# fizz_buzz result
def fizz_buzz(i, predictions):
    return [str(i), "fizz", "buzz", "fizzbuzz"][predictions]
final_answer = np.vectorize(fizz_buzz)(np.arange(1, 101),
                      sess.run(tf.argmax(prediction,1), {x:X_test,keep_prob:1}))
print(final_answer)
