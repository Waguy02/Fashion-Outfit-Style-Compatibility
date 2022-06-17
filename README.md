# Fashion outfit completion using bidirectionnal LSTM.

# Summary:
 - We aim to complete human fashion outfit using a neural network that 
learns style compatibilities of its items.
Our architecure is adapted from [Learning Fashion Compatibility with Bidirectional LSTMs](https://arxiv.org/abs/1707.05691).
and comprises 3 main components:
* A Cnn Encoder  : Resnet18 pretrained on ImageNet
* A Single layer BiLSTM 
* A MLP decoder with 2 hidden layers

- We perform training and evalutation of the [poylvore](https://github.com/xthan/polyvore-dataset) dataset 


By @waguy02



