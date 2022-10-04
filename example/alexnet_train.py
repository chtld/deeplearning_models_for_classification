import imp
import os
import sys
from traceback import print_tb
sys.path.insert(0, os.path.dirname(os.getcwd()))

from torchsummary import summary
from d2l import torch as d2l

from models.alexnet import AlexNet
from utils.d2l import train_ch6

net = AlexNet()

summary(net, input_size=[(1, 224, 224)], batch_size=1, device='cpu')

batch_size = 128

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)

lr, num_epochs = 0.01, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

d2l.plt.show()