__author__ = 'honore-nintunze'

import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
import numpy as np

############# Usefull functions #################
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


#################### Defining size of input and output ####################
# Define the size of the input and of the output
# entree
x = tf.placeholder(tf.float32, shape=[None, 784])
# sortie
y_ = tf.placeholder(tf.float32, shape=[None, 10])



#################### Building the DNN #####################
########## Declaring weights of layer 1 and 2 ##########
# Layer 1
# 784 entrees avec une couche intermediaire avec 1 neurone
W1 = weight_variable([784,1])
b1 = bias_variable([1])

# on rajoute la deuxieme couche
# Layer 2
# 1 neurone avec 10 neurones a la sortie
W2 = weight_variable([1,10])
b2 = bias_variable([10])

########## Building the intermediate error ##########
# Layer 1
a1 = tf.matmul(x,W1) + b1

# Layer 2
a2 = tf.matmul(a1,W2) + b2

# sortie du reseaux
y_out = tf.nn.softmax(a2)



#################### Defining the loss #####################
# Loss cross entropy
cross_entropy = - tf.reduce_sum(y_ * tf.log(y_out))
# with a L2 regularization
regularizer = tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1)
LAMBDA = 0.0001

loss = cross_entropy# + LAMBDA*regularizer



#################### Training method #####################
# Training with gradient descent
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)



#################### Creating a session and initializing variables #####################
# Creating a session
sess = tf.Session()
# Initialize variables
sess.run(tf.initialize_all_variables())



#################### Prediction error #####################
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# Gradient descent with minibatch of size 100
list_err = []
list_err_test = []
list_iteration = []



#################### Optimization #####################
for i in range(20000):
    # Creating the mini-batch
    batch_xs, batch_ys = mnist.train.next_batch(20)
    # running one step of the optimization method on the mini-batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i%100 == 0:
        # train error computation
        acuracy = sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
        print "###################################################"
        print "step %d, training err %g"%(i, 1-acuracy)

acuracy_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print "test accuracy %g"%(acuracy)
# GOT "test accuracy 0.354527", quelques secondes à s'executer
