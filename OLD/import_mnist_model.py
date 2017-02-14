import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.Session()
new_saver = tf.train.import_meta_graph('my-model.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.get_collection('vars')
for v in all_vars:
    v_ = sess.run(v)
    print(v_)

print len(v_)

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
