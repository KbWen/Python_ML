# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/word2vec
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
#
X = tf.placeholder(tf.int32, shape=[25])
# [batch_size, 1] for nn.nce_loss
Y_ = tf.placeholder(tf.int32, shape=[25, 1])
# sentences
sentences = ["They are like cats and dogs",
            "cats and dogs are friends",
            "she have cats and dogs"
            "cats and dogs are fighting"
            "cats are sensitive",
            "dog are loyal",
            "cats are liquid",
            "cats are wonderful",
            "dogs are barking",
            "cats are furry"
            "dogs are cute"
            "there have cats and dogs"
            "It's raining cats and dogs",
            "dogs and cats like baths"
            "I love cats and dogs",
            "we all love cats and dogs",
            "he likes cats",
            "she loves dogs",
            "everyone loves cats and dogs"]

def build_dataset(sent):
    # [(most count word1, n1),(second word2, n2)]
    count_word = collections.Counter(" ".join(sent).split()).most_common()
    rdictionary = [i[0] for i in count_word] #word
    dictionary = {w: i for i, w in enumerate(rdictionary)} #id
    data = [dictionary[word] for word in " ".join(sent).split()]
    # The actual code for this tutorial is very short
    # ([the, code], actual), ([actual, for], code),  ...
    cbow_pairs = []
    for i in range(1, len(data)-1) :
        cbow_pairs.append([[data[i-1], data[i+1]], data[i]])
    # skip-gram pairs
    # (actual, the), (actual, code), (code, actual), (brown, for), ...
    len_dic = len(dictionary)
    sgp = [];
    for i in cbow_pairs:
        sgp.append([i[1], i[0][0]])
        sgp.append([i[1], i[0][1]])
    return len_dic, sgp, rdictionary

len_dic, sgp, rdictionary = build_dataset(sentences)

# y = xw +b
W = tf.Variable(tf.random_uniform([len_dic, 2],-1.0, 1.0))
b = tf.Variable(tf.zeros([len_dic]))
# variables for the NCE loss
embeddings = tf.Variable(tf.random_uniform([len_dic, 2], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, X)
# tf.nn.nce_loss(weights, biases, inputs, labels, ...)
# negative samples
loss = tf.reduce_mean(tf.nn.nce_loss(W, b, Y_, embed, 12, len_dic))
train = tf.train.GradientDescentOptimizer(0.4).minimize(loss)

def generate_batch(size):
    assert size <= len(sgp)
    x_data = []
    y_data = []
    r = np.random.choice(range(len(sgp)), size, replace=False)
    for i in r:
        x_data.append(sgp[i][0])  # n dim
        y_data.append([sgp[i][1]])  # n, 1 dim
    return x_data, y_data

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        batch_inputs, batch_labels = generate_batch(25)
        _, loss_val = sess.run([train, loss], {X: batch_inputs, Y_: batch_labels})
        if i % 10 == 0:
          print("Loss :", i, loss_val) # loss
    # normalize
    final_embeddings = embeddings.eval()

for i, l in enumerate(rdictionary[:10]):
    x, y = final_embeddings[i,:]
    plt.scatter(x, y)
    plt.annotate(l, xy=(x, y), xytext=(5, 2),
        textcoords='offset points', ha='right', va='bottom')
plt.savefig("tf_word2vec.png")
