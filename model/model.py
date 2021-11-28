import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from torch.nn import Module
from torch.nn import Sequential

from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import BatchNorm2d
from torch.nn import Linear
from torch.nn import LSTM

from torch.nn import ReLU, Sigmoid
from torch.nn import BCEWithLogitsLoss

from database import OnsetsFramesVelocity, output_path
from transformer import Transformer

import torch.nn.functional as F
from torch import nn


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features //
                      16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16,
                      output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) *
                      (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsBaseline(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()
        model_size = model_complexity * 16

        self.conv = ConvStack(input_features, model_size)
        self.rnn = LSTM(model_size, model_size//2,
                        batch_first=True, bidirectional=True)
        self.fc = Linear(model_size, output_features)

    def forward(self, x, y):
        x = self.conv(x)
        x = self.rnn(x)[0]
        x = nn.ReLU()(x)
        x = self.fc(x)
        return x


class TOnly(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()


if __name__ == '__main__':
    dataset = OnsetsFramesVelocity(output_path)
    val_dataset = OnsetsFramesVelocity(output_path, split='val')
    model = OnsetsBaseline(229, 88)
    model.cuda()
    # print number of parameters

    dataset.train_split(model, split='frames', epochs=12, batch_size=8,
                        lr=6e-4, validation_data=val_dataset, save_path='frames_baseline.pt')
