from ctypes import resize
import sys
import os
sys.path.insert(0, os.path.dirname(os.getcwd()))

import torch
from torchsummary import summary
from d2l import torch as d2l
from models.vggnet import VGGNet
from utils.d2l import train_ch6



conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]

net = VGGNet(conv_arch, 1, 10)

summary(net, input_size=[(1, 224, 224)], batch_size=1, device='cpu')

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size, resize=224)

lr, num_epochs = 0.05, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, 'cpu')

d2l.plt.show()