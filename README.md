# learning-tensorflow

Repo containing code from some of the TensorFlow tutorials as well as some custom test code.

## Contents 

#### cifar10
*Description*: Contains files from the tutorial found [here](https://www.tensorflow.org/tutorials/deep_cnn#cifar-10_model).

#### cnn  
*Description*: Training and testing a convolutional neural network based on the MNIST dataset.  
`save_advanced_model.py` Trains the cnn based on the MNIST dataset (takes about 30 minutes to train). Saves the resulting variables to "variables".  
`test_advanced_model.py` Imports images from "image_database" and tests them against the saved cnn (tests instantaneously because the trained network is saved).  
`image_database` .png images used to test the convolutional neural network.
`MNIST_data` Dataset used to train the cnn.  
`variables` Saved variables from training the cnn.  

#### image_testing  
*Description*: Contains files used to test image importing and reconstructing MNIST images.

#### mnist_examples  
*Description*: Contains files from the two MNIST tutorials -- [softmax](https://www.tensorflow.org/get_started/mnist/beginners) and [cnn](https://www.tensorflow.org/get_started/mnist/pros).

#### simple_examples
*Description*: Contains examples of some of the basic features of TensorFlow
