import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my-model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    train_step = tf.get_collection('train_step')[0]
    # training (Using stochastic gradient descent)
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
