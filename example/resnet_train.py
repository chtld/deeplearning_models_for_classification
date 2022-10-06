from ctypes import resize
import os 
import sys 
sys.path.insert(0, os.path.dirname(os.getcwd()))

import torch.nn as nn
from d2l import torch as d2l
from models.resnet import ResNet
from utils.d2l import train_ch6
from torchsummary import summary

net = ResNet(1, 10)
summary(net, input_size=[(1, 224, 224)], batch_size=1, device='cpu')

lr, num_epochs, batch_size = 0.1, 10, 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()


