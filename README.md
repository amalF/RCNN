**Recurrent Convolutional Neural Network for Object Recognition using TF2.0**

*Before starting*
This code was tested using Python 3.6 and Tensorflow 2.0
All the dependencies are in the requirement.txt file

*Training*
As a baseline, I implemented the WCNN described in the paper.
Two types of normalization are tested for the recurrent convolution layer
: batch normalization and local response normalization.
In the experiments that I've done, I used a fixed learning rate, dropout and
batch normalization.

I trained the networks for 100 epochs.



