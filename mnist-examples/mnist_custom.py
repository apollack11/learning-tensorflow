import tensorflow as tf
import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# pull in testImage and convert to mnist format
testImage = cv2.imread("test_images/8.png")

testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
testImage = testImage.ravel()
testImage = testImage.astype(float)
for pixel,val in enumerate(testImage):
    if val == 255:
        testImage[pixel] = 0.
    else:
        testImage[pixel] = float(val)/255

# testImage = np.transpose(testImage)
testImage = np.reshape(testImage, (1,784))

print "Test Image Shape = ",testImage.shape

testLabel = np.array([0,0,0,0,0,0,0,0,1,0])
testLabel = np.reshape(testLabel, (1,10))
print "Test Label Shape = ",testLabel.shape

# MNIST is split into 3 parts
# Training Data: 55,000 data points (mnist.train)
# Test Data: 10,000 data points (mnist.test)
# Validation Data: 5,000 data points (mnist.validation)

# Each MNIST data point has 2 parts
# 1. Image of a handwirtten digit (x) (mnist.[set_name].images)
# 2. A label for this image (y) (mnist.[set_name].labels)

# each image is 28 by 28 pixels
# images are interpretted as an array of numbers
# can be flattened into a vector 28x28 = 784 numbers
# flattening the image can negatively impact some computer vision algorithms, but it doesn't matter for softmax regression

# mnist.train.images is a tensor with shape of [55000, 784] (number of images, number of pixels per image)
# mnist.train.labels is a tensor with shape of [55000, 10] where the label array is a vector with 1 in the position of the number it represents (e.g. [0,0,0,1,0,0,0,0,0,0] = 3)

# evidence_i = sum(W_i,j*x_j + b) where W is the weights, b is the bias
# y = softmax(evidence) where y is our predicted probabilities
# this procedure can be vectorized

# create a placeholder for input
# in this case we can input any number of MNIST images so None is used as the first element of the shape
x = tf.placeholder(tf.float32, [None, 784])

# create Vriables for the model parameters
W = tf.Variable(tf.zeros([784, 10])) # weights
b = tf.Variable(tf.zeros([10])) # biases

# implement the model
# multiply x and W using matmul
# then add b
# then apply tf.nn.softmax
y = tf.nn.softmax(tf.matmul(x, W) + b)

# need to figure out the cost
# one common function for this is "cross-entropy"
# H(y) = -sum(y' * log(y)) where y is our predicted probability distribution and y' is the true distribution (the vector with digit labels e.g. [0,0,0,1,0,0,0,0,0,0])

# to implement cross-entropy we need to add a placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
# implement the cross-entropy function
# tf.log compute log of each element of y
# then log(y) is multiplied by y_
# then tf.reduce_sum adds the elements in the second dimension of y (because of the reduction_indices=[1] parameter)
# finally reduce_mean computes the mean over all examples
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

### NOTE: the source code does not use this formulation, instead they use:
# y = tf.matmul(x, W) + b
# y_ = tf.placeholder(tf.float32, [None, 10])
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# TensorFlow automatically does backpropagation
# Then it can apply your choice of optimization algorithm
# minimize cross_entropy using gradient descent with a learning rate of 0.5
# NOTE: tweak this line to use other optimizers (https://www.tensorflow.org/api_docs/python/train/#optimizers)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# operation to initilize Variables
init = tf.global_variables_initializer()

# now launch the model in a Session
sess = tf.Session()
sess.run(init)

# training (Using stochastic gradient descent)
# we'll run the training step 1000 times
# for each loop, we get a "batch" of 100 random data points from our training set
# we run train_step feeding the batches of data to replace the placeholders
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# EVALUATING OUR MODEL
# checks to see if the maximum value along the first dimension of y is the same as in y_
# this gives us a list of booleans [True,False,True,True,etc.]
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# convert list of booleans to floating point numbers [1,0,1,1,etc.]
# take the mean of these predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print the accuracy of the model on our test data
print "Test Accuracy"
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print "Custom Test Accuracy"
print(sess.run(accuracy, feed_dict={x: testImage, y_: testLabel}))

print "Custom Test Prediction"
print(sess.run(tf.argmax(y,1)[0], feed_dict={x: testImage, y_: testLabel}))

print "Custom Test Actual"
print np.argmax(testLabel,1)[0]

print("weights:", sess.run(W))
print("biases:", sess.run(b))

# NOTE: this model will yield an accuracy of about 92%, this is not very good, simple improvements can improve our results to over 97%, and the best models can get over 99.7% accuracy
