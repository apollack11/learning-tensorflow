import tensorflow as tf
import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# pull in testImage and convert to mnist format
testImage = cv2.imread("test_images/1.png")

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

testLabel = np.array([0,1,0,0,0,0,0,0,0,0])
testLabel = np.reshape(testLabel, (1,10))
print "Test Label Shape = ",testLabel.shape

# create a placeholder for input
x = tf.placeholder(tf.float32, [None, 784])

# create Vriables for the model parameters
W = tf.Variable(tf.zeros([784, 10])) # weights
b = tf.Variable(tf.zeros([10])) # biases

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

print "Custom Test Accuracy"
print(sess.run(accuracy, feed_dict={x: testImage, y_: testLabel}))

print "Custom Test Prediction"
print(sess.run(tf.argmax(y,1)[0], feed_dict={x: testImage, y_: testLabel}))

print "Custom Test Actual"
print np.argmax(testLabel,1)[0]


# NOTE: this model will yield an accuracy of about 92%, this is not very good, simple improvements can improve our results to over 97%, and the best models can get over 99.7% accuracy
