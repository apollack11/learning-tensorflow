import tensorflow as tf

# create a Variable, intialized to 0
state = tf.Variable(0, name="counter")

# create an Op to add one to 'state'
one = tf.constant(1)
new_value = tf.add(state, one)
# does not actuall perform the assignment until run() executes the expression
update = tf.assign(state, new_value)

# Variables must be initialized by running an 'init' Op after having launched the graph
# add the 'init' Op to the graph
init_op = tf.global_variables_initializer()

# launch the graph and run the ops
with tf.Session() as sess:
    # run the 'init' op
    sess.run(init_op)
    # print the initial value of 'state'
    print(sess.run(state))
    # run the op that updates 'state' and print 'state'
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
