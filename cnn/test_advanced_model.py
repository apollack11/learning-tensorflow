import tensorflow as tf
import os
import cv2
import numpy as np
import time
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

t0 = time.time()

directory = "image_database"

files = []
for f in os.listdir(directory):
    if f.endswith('.png'):
        files.append(f)

num_images = len(files)

images = np.zeros((num_images, 784))
labels = np.zeros((num_images, 10))

for i,filename in enumerate(files):
    print i,directory + '/' + filename
    # import image and convert to numpy array of proper size
    image = cv2.imread(directory + '/' + filename)
    # image = cv2.imread(os.path.join(directory, filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.ravel()
    image = image.astype(float)
    image = 1 - image/255
    image = np.reshape(image, (1,784))
    # add image to images
    images[i] = image

    # import label
    digit = int(filename[0])
    image_label = np.array([0,0,0,0,0,0,0,0,0,0])
    image_label[digit] = 1
    image_label = np.reshape(image_label, (1,10))
    # add image_label to labels
    labels[i] = image_label

sess = tf.InteractiveSession()

# create a placeholder for input
x = tf.placeholder(tf.float32, [None, 784])

# to implement cross-entropy we need to add a placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

# NOTE: info on convolutional neural networks: http://cs231n.github.io/convolutional-networks/

# create functions to initialize weights with a slightly positive initial bias to avoid "dead neurons"
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# first convolutional layer
# 32 features for each 5x5 patch
# weight tensor will have a shape of [5,5,1,32]
# first two dimensions are patch size (5x5)
# next is the number of input channels (1)
# last is the number of output channels (32)
W_conv1 = weight_variable([5,5,1,32], 'W_conv1')
b_conv1 = bias_variable([32], 'b_conv1')

# reshape x to a 4d tensor, 2nd and 3rd dimensions are image width and height, final dimension is the number of color channels
x_image = tf.reshape(x, [-1,28,28,1])

# convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
# 64 features for each 5x5 patch
W_conv2 = weight_variable([5,5,32,64], 'W_conv2')
b_conv2 = bias_variable([64], 'b_conv2')

# convolve the result of h_pool1 with the weight tensor, add the bias, apply the ReLU function, and finally max pool
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
# image size has been reduced to 7x7 so we will add a fully-connected layer with 1024 neurons
W_fc1 = weight_variable([7*7*64, 1024], 'W_fc1')
b_fc1 = bias_variable([1024], 'b_fc1')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
# to reduce overfitting, we apply dropout before the readout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
W_fc2 = weight_variable([1024, 10], 'W_fc2')
b_fc2 = bias_variable([10], 'b_fc2')

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# MAKING A PREDICTION BASED ON OUR SAVED MODEL
# checks to see if the maximum value along the first dimension of y is the same as in y_
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

# convert list of booleans to floating point numbers [1,0,1,1,etc.]
# take the mean of these predictions
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print the accuracy of the model on our test data
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "variables/advanced_model.ckpt")
    # display accuracy of entire test set
    # print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    # print the accuracy of the model on our test data
    print "Custom Test Accuracy"
    print(accuracy.eval(feed_dict={x: images, y_: labels, keep_prob: 1.0}))

    print "Custom Test Prediction"
    print(sess.run(tf.argmax(y_conv,1), feed_dict={x: images, y_: labels, keep_prob: 1.0}))

    print "Custom Test Actual"
    print np.argmax(labels,1)

    result = sess.run(y_conv, feed_dict={x: images, y_: labels, keep_prob: 1.0})
    print "y_conv: ",result

# NOTE: this model will yield an accuracy of about 92%, this is not very good, simple improvements can improve our results to over 97%, and the best models can get over 99.7% accuracy

t1 = time.time()

print "Time to run:",t1-t0,"seconds"
