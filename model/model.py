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

<<<<<<< HEAD
=======
import torch
>>>>>>> 77a930e2a926ced798fca648dcb7653211924f2d
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
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
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
        self.rnn = LSTM(model_size, model_size//2, batch_first=True, bidirectional=True)
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
    model = OnsetsBaseline(229,88)
    model.cuda()
    # print number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    dataset.train(model, split='frames', epochs=500, batch_size=8, lr=1e-2)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # criterion = BCEWithLogitsLoss()
    
    # for i, batch in enumerate(loader):
    #     spec, onsets = batch['real'].cuda(), batch['onsets'].cuda()
    #     spec = spec.transpose(1,2)
    #     onsets = onsets.transpose(1,2)
    #     break
    
    # losses = np.empty(5000)
    # maxes = np.empty(5000)
    # for epoch in range(5000):
    #     out = model(spec)
    #     loss = criterion(out, onsets)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     losses[epoch] = loss.item()
    #     maxes[epoch] = torch.max(out).item()
    #     if epoch % 100 == 0:
    #         print(np.mean(losses[max(0, epoch-100):epoch]))
    # # create the two axes on left and right side
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # # plot the loss
    # ax1.plot(losses)
    # ax1.set_title('Loss')
    # # plot the maxes
    # ax2.plot(maxes)
    
    
