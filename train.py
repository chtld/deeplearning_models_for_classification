'''
相较于简单版本的训练脚本 train_sample 增添了以下功能：
1. 使用argparse类实现可以在训练的启动命令中指定超参数
2. 可以通过在启动命令中指定 --seed 来固定网络的初始化方式，以达到结果可复现的效果
3. 可以通过在启动命令中指定 --tensorboard 来进行tensorboard可视化, 默认不启用。
4. 可以通过在启动命令中指定 --model 来选择使用的模型
   注意，使用tensorboad之前需要使用命令 "tensorboard --logdir= log_path"来启动，结果通过网页 http://localhost:6006/'查看可视化结果
5. 使用了一个更合理的学习策略：在训练的第一轮使用一个较小的lr（warm_up），从第二个epoch开始，随训练轮数逐渐减小lr。
6. 使用amp包实现半精度训练，在保证准确率的同时尽可能的减小训练成本

--model 可选的超参如下：
lenet alexnet vggnet ninnet googlenet

训练命令示例： # python train.py --model alexnet --num_classes 5
'''
import os 
import sys 
import json 
import argparse
import shutil
import random
import numpy as np
import torch 
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=10, help='the number of classes')
parser.add_argument('--epochs', type=int, default=50, help='the number of traing epoch')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size for traing')
parser.add_argument('--lr', type=float, default=0.0002, help='start learning rate')
parser.add_argument('--lrf', type=float, default=0.0001, help='final learning rate')
parser.add_argument('--seed', default=False, action='store_true', help='fix the initialization of parameters')
parser.add_argument('--tensorboard', default=False, action='store_true', help=' use tensorboard for visualization') 
parser.add_argument('--data_path', type=str, default="/data/flowers")
parser.add_argument('--model', type=str, default="vgg", help=' select a model for training') 
parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
opt = parser.parse_args()

if opt.seed:
    def seed_torch(seed=7):
        random.seed(seed) # Python random module.	
        os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed) # Numpy module.
        torch.manual_seed(seed)  # 为CPU设置随机种子
        torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        # 设置cuDNN：cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。如果需要保证可重复性，可以使用如下设置:
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        # 实际上这个设置对精度影响不大，仅仅是小数点后几位的差别。所以如果不是对精度要求极高，其实不太建议修改，因为会使计算效率降低。
        print('random seed has been fixed')
    seed_torch() 

def main(args):
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if opt.tensorboard:
        # 存放tensorboard显示的数据的绝对路径
        log_path = os.path.join('/data/tensorboard', args.model)
        print('Start Tensorboard with "tensorboard --logdir=%s"' % (log_path))
        

if __name__ == '__main__':
    main(opt)