import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# create a placeholder for input
x = tf.placeholder(tf.float32, [None, 784])

# create Vriables for the model parameters
W = tf.Variable(tf.zeros([784, 10])) # weights
tf.add_to_collection('vars', W)
b = tf.Variable(tf.zeros([10])) # biases

# implement the model
y = tf.matmul(x, W) + b

# add a placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
# implement the cross-entropy function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

saver = tf.train.Saver([W])

# NOTE: tweak this line to use other optimizers (https://www.tensorflow.org/api_docs/python/train/#optimizers)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.add_to_collection('train_step', train_step)

# now launch the model in a Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training (Using stochastic gradient descent)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# export model
saver.save(sess, 'my-model')
