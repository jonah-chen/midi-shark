import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=(1,stride), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1,stride), bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or stride != 1 else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class mResNet50(nn.Module):
    def __init__(self, out_features=768):
        super(mResNet50, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((1,2))
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.avgpool = nn.AvgPool2d(1)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.mean(x, dim=3)
        x = x.transpose(1,2)
        return x


from common import ConvStack, SplitModule
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from database import OnsetsFramesVelocity, output_path
from metrics import *
from time import time
from tqdm import tqdm


class ResNetLSTM(SplitModule):
    def __init__(self, split='onsets'):
        super().__init__(split)
        if split not in ['onsets', 'offsets', 'frames', 'velocities']:
            raise ValueError(
                'split must be either onsets, offsets, frames, or velocities')

        self.conv = mResNet50()
        self.rnn = nn.LSTM(512, 256,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, 88)
        
    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = self.rnn(x)[0]
        x = nn.ReLU()(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    dataset = OnsetsFramesVelocity(output_path)
    val_dataset = OnsetsFramesVelocity(output_path, split='val')
    model = ResNetLSTM('frames')
    model.cuda()
    model.fit(dataset, validation_data=val_dataset, epochs=12, lr=4e-3)