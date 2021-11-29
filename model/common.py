import numpy as np
import math
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from time import time
from tqdm import tqdm
from metrics import *

class SplitModule(nn.Module):
    def __init__(self, split='onsets'):
        super().__init__()
        if split not in ['onsets', 'offsets', 'frames', 'velocities']:
            raise ValueError(
                f'Split must be one of onsets, offsets, frames, or velocities. Got {split}.')
        self.split = split
    
    def fit(
        self,
        x,
        batch_size=4,
        shuffle=True,
        num_workers=24,
        epochs=1,
        verbose=True,
        device='cuda',
        loss_fn=BCEWithLogitsLoss,
        optimizer=Adam,
        lr=0.0006,
        validation_data=None,
        save_path='dummy'
    ):
        """
        Train the split of the model on the dataset.

        Args:
            model (nn.Module): The model to train.
            split (str): The split to train on. Must be either 'onsets', 
            'offsets', 'frames', or 'velocities'. Defaults to 'onsets'.
            batch_size (int): The batch size to use. Defaults to 4.
            shuffle (bool): Whether to shuffle the dataset. Defaults to True.
            num_workers (int): Number of workers to use. Defaults to 24.
            epochs (int): Number of epochs to train for. Defaults to 1.
            verbose (bool): Whether to print out extra details. Defaults to True.
            device (str): The device to train on. Defaults to 'cuda'. Must change to 'cpu' if no GPU is available.
            loss_fn (nn.Module): The loss function to use. Defaults to BCEWithLogitsLoss.
            optimizer (nn.Module): The optimizer to use. Defaults to Adam.
            lr (float): The learning rate to use. Defaults to 0.0006.
            validation_data (tuple): The validation data to use. Defaults to None, meaning validation is not performed.
            save_path (str): The path to save the model to. Defaults to 'dummy'.
            if save_path is None, the model is not saved.
        """
        train_start_time = int(time()) // 1000000
        
        if verbose:
            print(sum(p.numel()
                      for p in self.parameters() if p.requires_grad), 'parameters')

        data_loader = DataLoader(
            x, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        criterion = loss_fn()
        optim = optimizer(self.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss, epoch_P, epoch_R, epoch_F1, epoch_min, epoch_max = 0, 0, 0, 0, 0, 0

            # start training model
            self.train()
            data_iter = tqdm(enumerate(data_loader), ascii=True, total=len(
                data_loader)) if verbose else enumerate(data_loader)
            for i, batch in data_iter:
                # get data
                spec = batch['real'].to(device).transpose(1, 2)
                spec = (spec + 40)/40

                truth = batch[self.split].to(device).transpose(1, 2)

                # forward pass
                out = self(spec)
                loss = criterion(out, truth)

                # backward pass
                optim.zero_grad()
                loss.backward()
                optim.step()

                # calculate precision, recall and f1 score
                pred = out > 0
                truth = truth > 0
                with torch.no_grad():
                    P = precision(truth, pred)
                    R = recall(truth, pred)

                    # update loss, precision, recall, f1 score and min/max
                    epoch_loss += loss.item()
                    epoch_P += P
                    epoch_R += R
                    epoch_F1 += 2 * P * R / (P + R + 1e-8)
                    epoch_min += torch.min(out).item()
                    epoch_max += torch.max(out).item()

                if verbose:
                    data_iter.set_description(
                        f'Epoch {epoch + 1}/{epochs} - Loss: {1e3*epoch_loss/(i+1):.1f} - P: {100*epoch_P/(i+1):.2f}% - R: {100*epoch_R/(i+1):.2f}% - F1: {100*epoch_F1/(i+1):.2f}% - [{epoch_min/(i+1):.2f},{epoch_max/(i+1):.2f}]')
                else:
                    print(
                        f'Epoch {epoch + 1}/{epochs} - Loss: {1e3*epoch_loss/(i+1):.1f} - F1: {100*epoch_F1/(i+1):.2f}%')

            # perform validation if validation data is not None
            if validation_data:
                result = self.val_split(
                    validation_data,
                    split=split,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    device=device
                )

                if verbose:
                    print(
                        f"Validation -  P: {result['P']:.2f}% - R: {result['R']:.2f}% - F1: {result['F1']:.2f}% - [{result['min']:.2f},{result['max']:.2f}]")
                else:
                    print(f"Validation - F1: {result['F1']:.2f}%")

            # save model if save_path is not None, save model
            if save_path:
                torch.save(self.state_dict(), f"{save_path}_{train_start_time}.pt")
    

    def val_split(
        self,
        val_x,
        split='onsets',
        batch_size=4,
        shuffle=True,
        num_workers=24,
        device='cuda'
    ):
        """
        Validate the model on the dataset. This generally should not be used for 
        training split. 

        Args:
            model (nn.Module): The model to validate.
            split (str): The split to validate on. Must be either 'onsets', 
            'offsets', 'frames', or 'velocities'. Defaults to 'onsets'.
            batch_size (int): The batch size to use. Defaults to 4.
            shuffle (bool): Whether to shuffle the dataset. Defaults to True.
            num_workers (int): Number of workers to use. Defaults to 24.
            device (str): The device to validate on. Defaults to 'cuda'. Must change to 'cpu' if no GPU is available.

        Returns:
            dict: A dictionary containing the precision, recall, f1 score, min, and max.
        """
        data_loader = DataLoader(
            val_x, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        P, R, F1, min_, max_ = 0, 0, 0, 0, 0

        self.eval()

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                # get data
                spec = batch['real'].to(device).transpose(1, 2)
                spec = (spec + 40)/40

                truth = batch[self.split].to(device).transpose(1, 2)

                # forward pass
                out = self(spec)
                pred = out > 0
                truth = truth > 0
                # calculate precision, recall
                p = precision(truth, pred)
                r = recall(truth, pred)
                # accumulate precision, recall, f1 score, min, max
                P += p
                R += r
                F1 += 2 * p * r / (p + r + 1e-8)
                min_ += torch.min(out).item()
                max_ += torch.max(out).item()

        return {'P': 100*P/len(data_loader),
                'R': 100*R/len(data_loader),
                'F1': 100*F1/len(data_loader),
                'min': min_/len(data_loader),
                'max': max_/len(data_loader)}



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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len, device='cuda').unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device='cuda') * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model, device='cuda')
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1), :]
