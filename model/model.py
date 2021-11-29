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
            nn.MaxPool2d((1, 2))
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) *
                      (input_features // 4), output_features),
            nn.BatchNorm1d(862),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x

import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, device='cuda').unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device='cuda') * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model, device='cuda')
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = ConvStack(ntoken, d_model)#nn.Embedding(ntoken, d_model)
        self.d_model = math.sqrt(d_model)
        self.decoder = nn.Linear(d_model, 88)

    def forward(self, src: Tensor) -> Tensor:
        src = self.encoder(src) * self.d_model
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, torch.zeros(src.size(1),src.size(1),device='cuda',requires_grad=False))
        # output = self.transformer_encoder(src, torch.zeros(src.size(1), src.size(1), device='cuda'))
        src = nn.ReLU()(src)
        return self.decoder(src)

class OnsetsBaseline(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()
        model_size = model_complexity * 16

        self.conv = ConvStack(input_features, model_size)
        self.rnn = LSTM(model_size, model_size//2,
                        batch_first=True, bidirectional=True)
        self.fc = Linear(model_size, output_features)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.rnn(x)[0]
        x = nn.ReLU()(x)
        x = self.fc(x)
        return x
    

class OnsetsOffsetsFramesBaseline(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()
        model_size = model_complexity * 16

        self.onsets_model = OnsetsBaseline(input_features, output_features, model_complexity)
        self.offsets_model = OnsetsBaseline(input_features, output_features, model_complexity)

        self.conv = ConvStack(input_features, model_size)
        self.rnn = LSTM(model_size+output_features+output_features, model_size//2, batch_first=True, bidirectional=True)
        self.fc = Linear(model_size, output_features)


    def forward(self, x):
        onsets = self.onsets_model(x)
        offsets = self.offsets_model(x)
        x = self.conv(x)
        x = torch.cat((x, onsets.detach(), offsets.detach()), dim=2)
        x = self.rnn(x)[0]
        x = nn.ReLU()(x)
        x = self.fc(x)
        return onsets, offsets, x
    


if __name__ == '__main__':
    dataset = OnsetsFramesVelocity(output_path)
    val_dataset = OnsetsFramesVelocity(output_path, split='val')
    model = TransformerModel(229, 512, 16, 512, 8)
    model.cuda()

    dataset.train_split(model, split='frames', validation_data=val_dataset, epochs=12, lr=6e-5)
    # dataset.train_oof(model, epochs=15, batch_size=4, lr=6e-4, validation_data=val_dataset, save_path='oof_baseline.pt')
