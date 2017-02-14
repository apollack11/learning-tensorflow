import tensorflow as tf
import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# MNIST is split into 3 parts
# Training Data: 55,000 data points (mnist.train)
# Test Data: 10,000 data points (mnist.test)
# Validation Data: 5,000 data points (mnist.validation)

# create a placeholder for input
x = tf.placeholder(tf.float32, [None, 784])

# create Vriables for the model parameters
W = tf.Variable(tf.zeros([784, 10]), name='weights') # weights
b = tf.Variable(tf.zeros([10]), name='biases') # biases

# implement the model
y = tf.matmul(x, W) + b

# add a placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
# implement the cross-entropy function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# NOTE: tweak this line to use other optimizers (https://www.tensorflow.org/api_docs/python/train/#optimizers)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# operation to initilize Variables
init = tf.global_variables_initializer()

# now launch the model in a Session
sess = tf.Session()
sess.run(init)

# training (Using stochastic gradient descent)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# EVALUATING OUR MODEL
# checks to see if the maximum value along the first dimension of y is the same as in y_
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# convert list of booleans to floating point numbers [1,0,1,1,etc.]
# take the mean of these predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print the accuracy of the model on our test data
print "Test Accuracy"
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print("weights:", sess.run(W))
print("biases:", sess.run(b))

saver = tf.train.Saver()

save_path = saver.save(sess, "my_net2/save_variables.ckpt")
print("Save to path: ", save_path)

# NOTE: this model will yield an accuracy of about 92%, this is not very good, simple improvements can improve our results to over 97%, and the best models can get over 99.7% accuracy
