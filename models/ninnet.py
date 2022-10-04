import torch.nn as nn 


class NiNNet(nn.Module):
    def __init__(self, input_channel=1, num_classes=10):
        super(NiNNet, self).__init__()
        self.features_and_classifier = nn.Sequential(
            self.nin_block(input_channel, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            self.nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            self.nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            self.nin_block(384, num_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
    def forward(self, x):
        return self.features_and_classifier(x)
    
    def nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )