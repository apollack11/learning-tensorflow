# CIFAR10 Example

## Model Architecture  
The model consists of alternating convolutions and nonlinearities. These layers are followed by fully connected layers leading into a softmax classifier. The model follows the architecture described by Alex Krizhevsky, with a few differences in the top few layers.  

This model achieves a peak performance of about 86% accuracy within a few hours of training time on a GPU.  

## Model Inputs  
The input part of the model is built by the functions inputs() and distorted_inputs() which read images from the CIFAR-10 binary data files. These files contain fixed byte length records, so we use tf.FixedLengthRecordReader.  

The images are processed as follows:
- They are cropped to 24x24 pixels, centrally for evaluation or randomly for training.  
- They are approximately whitened to make the model insensitive to dynamic range.  

For training, we additionally apply a series of random distortions to artificially increase the data set size:
- Randomly flip the image from left to right  
- Randomly distort the image brightness  
- Randomly distort the image contrast  

Reading images from disk and distorting them can use a non-trivial amount of processing time. To prevent these operations from slowing down training, we run them inside 16 separate threads which continuously fill a TensorFlow queue.  

## Model Prediction  
The prediction part of the model is constructed by the interference() function which adds operations to compute the logits of the predictions. That part of the model is organized as follows:  
conv1: convolution and rectified linear activation  
pool1: max pooling  
norm1: local response normalization  
conv2: convolution and rectified linear activation  
norm2: local response normalization  
pool2: max pooling  
local3: fully connected layer with rectified linear activation  
local4: fully connected layer with rectified linear activation  
softmax_linear: linear transformation to produce logits  

## Model Training  
The usual method for training a network to perform N-way classification is multinomial logistic regression (aka. softmax regression). Softmax regression applies a softmax nonlinearity to the output of the network and calculates the cross-entropy between the normalized predictions and a 1-hot encoding of the label. For regularization, we also apply the usual weight decay losses to all learned variables. The objective function for the model is the sum of the cross entropy loss and all these weight decay terms, as returned by the loss() function.  

Can visualize it in TensorBoard with a `scalar_summary`  

We train the model using standard gradient descent algorithm with a learning rate that exponentially decays over time.  

The train() function adds the operations needed to minimize the objective by calculating the gradient and updating the learned variables. It returns an operation that executes all the calculations needed to train and update the model for one batch of images.  
