import tensorflow as tf

# BUILDING THE GRAPH
# 3 NODES: 2 contant() ops and 1 matmul() op
# creates a Constant that produces a 1x2 matrix
matrix1 = tf.constant([[3., 3.]])
# creates a Constant that produces a 2x1 matrix
matrix2 = tf.constant([[2.],[2.]])
# create a Matmul op that takes the product of the two matrices
product = tf.matmul(matrix1, matrix2)

# LAUNCHING THE GRAPH IN SESSION
# lanch the default graph
sess = tf.Session()

# call the session run() method passing in 'product'
# this will run all 3 ops needed for the computation
# it will run matmul() which in turn will run constant() for both constants
result = sess.run(product)
# prints result as a numpy array
print(result)

# closes the session
sess.close()
