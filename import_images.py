import os
import cv2
import numpy as np

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
