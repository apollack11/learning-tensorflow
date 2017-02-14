import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import cv2

testImage = cv2.imread("8.png")

testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
testImage = testImage.ravel()
testImage = testImage.astype(float)
for pixel,val in enumerate(testImage):
    if val == 255:
        testImage[pixel] = 0.
    else:
        testImage[pixel] = float(val)/255

testLabel = [0,0,0,0,0,0,0,0,1,0]

print mnist.test.images[0]
print len(mnist.test.images[0])
