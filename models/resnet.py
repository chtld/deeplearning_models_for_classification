from turtle import forward
import torch 
import torch.nn as nn 
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)
        

class ResNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(ResNet, self).__init__()
        b1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        b5 = nn.Sequential(*self.resnet_block(256, 512, 2))
        self.net = nn.Sequential(
            b1, b2, b3, b4, b5,
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)
    
    def resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk