import os 
import sys 
sys.path.insert(0, os.path.dirname(os.getcwd()))
from d2l import torch as d2l
from models.googlenet import GoogLeNet
from utils.d2l import train_ch6

net = GoogLeNet(1, 10)
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()
