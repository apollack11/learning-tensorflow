import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# takes in the MNIST dataset and converts an individual image to a readable format
# this is used to verify the formula for converting readable images into MNIST images

testImage = cv2.imread("test_images/7.png")

testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)

mnistImage = mnist.train.images[4300]

print mnistImage

for pixel,val in enumerate(mnistImage):
    mnistImage[pixel] = 255 - float(val)*255

mnistImage = np.reshape(mnistImage, (28,28))
mnistImage = mnistImage.astype('uint8')

print mnistImage

cv2.imshow('mnistImage',mnistImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
