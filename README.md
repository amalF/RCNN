# Recurrent Convolutional Neural Network for Object Recognition using TF2.0


## Before starting
This code was tested using Python 3.6 and Tensorflow 2.0
All the dependencies are in the requirement.txt file

## Training
As a baseline, I implemented the WCNN described in the paper.

Two types of normalization are tested for the recurrent convolution layer
: batch normalization and local response normalization.
In the experiments that I've done, I used a fixed learning rate, dropout and
batch normalization.

I trained the networks for 100 epochs.

## Why recurrent connections ?
In the visual cortex, recurrent conenctions are playing a major role in
learning. There are type types of recurrent connections : **local recurrent
conenctions** and **long range recurrent connections**. In ResNet, lon range
connections are added and skip connections show great impact on the
performance. RCNN is introducing local recurrent connections. A [recent
paper](https://papers.nips.cc/paper/7775-task-driven-convolutional-recurrent-models-of-the-visual-system.pdf) added both types of recurrent conenctions to a feedforward CNN to mimic the dynamic of visual system.

##Contact
amal.feriani@gmail.com

##Licence
MIT



