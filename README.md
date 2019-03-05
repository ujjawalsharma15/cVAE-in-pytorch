# Implementation of cVAE in pytorch
Implementation of convolutional VAE in pytorch

## Dataset 
MNIST dataset which consists of images of shape 1x28x28

## Implementation Details
Encoder-Decoder network consisting of conv and convtranspose layers respectively, activation function used is LeakyReLU instead of
sigmoid as suggested in original paper to tackle vanishing gradient problem.

Learning Rate = 2*e-05

Batch Size = 16

Epochs = 50

Optimizer = Adam with betas-(0.5,0.999)

## Results
Reconstructed Images from original images:


![img-1](https://user-images.githubusercontent.com/33577587/53810670-79e4a400-3f7d-11e9-890a-3d752d6a89a0.png)
![img-2](https://user-images.githubusercontent.com/33577587/53810709-8cf77400-3f7d-11e9-8515-a006f7d22a22.png)


Images constructed from sampled noise:


![img-3](https://user-images.githubusercontent.com/33577587/53810817-c16b3000-3f7d-11e9-8e56-cc4af25bf86d.png)
![img-4](https://user-images.githubusercontent.com/33577587/53810832-cc25c500-3f7d-11e9-919f-75822e9dce87.png)
![img-5](https://user-images.githubusercontent.com/33577587/53810841-d647c380-3f7d-11e9-91c5-80226d81f74b.png)


Loss Curve:-

![img-6](https://user-images.githubusercontent.com/33577587/53810919-13ac5100-3f7e-11e9-865b-6a8e42727431.png)
![img-7](https://user-images.githubusercontent.com/33577587/53810952-2888e480-3f7e-11e9-8fa4-757334e0e687.png)
