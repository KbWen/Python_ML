import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Model input and output
x_ = tf.placeholder(tf.float32,[None,1])
y_ = tf.placeholder(tf.float32,[None,1])
# define parameters and linear model
def new_layer(input_data, in_dim, hidden_units):
    global W, b
    W = tf.Variable(tf.random_normal([in_dim, hidden_units]))
    b = tf.Variable(tf.random_normal([1,hidden_units]))
    Y = tf.matmul(input_data, W) + b
    return Y

# activation function and hypothesis set
hidden_layer1 = tf.nn.tanh(new_layer(x_, 1, 10))
##hidden_layer2 = tf.nn.relu(new_layer(hidden_layer1,4,8))
predictions = new_layer(hidden_layer1 ,10 , 1)
# loss
loss = tf.reduce_mean(tf.square(y_ - predictions))
# optimizer
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# training data
x_train = np.linspace(-1,1,200)[:,np.newaxis]
noise = np.random.normal(0,0.09,x_train.shape)
y_train = np.power(x_train,4) - 2*np.power(x_train,3) +noise
# show training data
fig,ax = plt.subplots()
fig.set_tight_layout(True)
ax.scatter(x_train,y_train)
# training loop
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    sess.run(train, {x_:x_train,y_:y_train})
    if i %50==0:
        try:
            ax.lines.remove(lines[0])
        except:
            pass
        print(sess.run(loss, {x_:x_train, y_:y_train}))
        # show training line    #use color=(R,G,B)
        lines = ax.plot(x_train, sess.run(predictions, {x_:x_train}),color=(0.8,0.7,0.15),lw = 3)
        plt.pause(0.08)

plt.ion()
plt.draw()
# evaluate training accuracy
curr_W, curr_b, curr_loss, = sess.run([W, b, loss], {x_:x_train, y_:y_train})
print("W: %s \n b: %s \n loss: %s"%(curr_W, curr_b, curr_loss))
