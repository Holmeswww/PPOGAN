# CIFAR10 Experiments

The code is adapted from [here](https://github.com/biuyq/CT-GAN)

This code implements a GAN for CIFAR10 with a ResNet architecture in Pytorch. Inception/FID reporting is also integrated into the training process. Note that since IS differs between Pytorch and Tensorflow implementations, we call a Tensorflow backend to calculate IS for the sake of fair comparison.

The hyper-parameter inception score of around 8.6 on CIFAR 10.
## Usage

Usage
```
python main.py

```
