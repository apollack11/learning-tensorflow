import tensorflow as tf

# a feed temporarily replaces the output of an operation with a tensor value
# you supply feed data as an argument to a run() call
# most commonly, designate specific operations to be "feed" operations by using tf.placeholder() to create them
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
