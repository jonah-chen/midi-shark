from common import SplitModule, ConvStack, PositionalEncoding
import math
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from database import output_path, OnsetsFramesVelocity
from metrics import *
from torch.optim import Adam
from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss

class TransformerModel(SplitModule):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.0, split='onsets', bias=False):
        super().__init__(split)
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = ConvStack(ntoken, d_model, bias=bias)#nn.Embedding(ntoken, d_model)
        self.d_model = math.sqrt(d_model)
        self.decoder = nn.Linear(d_model, 88)

    def forward(self, src):
        src = self.encoder(src) * self.d_model
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, torch.zeros(src.size(1),src.size(1),device='cuda',requires_grad=False))
        src = nn.ReLU()(src)
        return self.decoder(src)

class TransformerOnlyModel(SplitModule):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.0, split='onsets'):
        super().__init__(split)
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(228)
        encoder_layers = TransformerEncoderLayer(228, 19, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = math.sqrt(d_model)
        self.decoder = nn.Linear(228, 88)

    def forward(self, src):
        src = src[:,:,:228]
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, torch.zeros(src.size(1),src.size(1),device='cuda',requires_grad=False))
        src = nn.ReLU()(src)
        return self.decoder(src)
    
class VelocityModel(SplitModule):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                nlayers: int, dropout: float = 0.2):
        super().__init__('velocities')
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = ConvStack(ntoken, d_model)
        self.d_model = math.sqrt(d_model)
        self.decoder = nn.Linear(d_model, 88)

    def forward(self, src):
        src = self.encoder(src) * self.d_model
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, torch.zeros(src.size(1),src.size(1),device='cuda',requires_grad=False))
        src = nn.ReLU()(src)
        src = self.decoder(src)
        return nn.Sigmoid()(src)

    def fit(
        self,
        x,
        batch_size=4,
        shuffle=True,
        num_workers=24,
        epochs=1,
        verbose=True,
        device='cuda',
        loss_fn=modified_mse,
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
        train_start_time = int(time()) % 1000000
        
        if verbose:
            print(sum(p.numel()
                      for p in self.parameters() if p.requires_grad), 'parameters')

        data_loader = DataLoader(
            x, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        criterion = loss_fn()
        optim = optimizer(self.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss, ee1, ee3, ee10 = 0, 0, 0, 0

            # start training model
            self.train()
            data_iter = tqdm(enumerate(data_loader), ascii=True, total=len(
                data_loader)) if verbose else enumerate(data_loader)
            for i, batch in data_iter:
                # get data
                spec = batch['real'].to(device).transpose(1, 2)
                spec = (spec + 40)/40

                truth = batch[self.split].to(device).transpose(1, 2)
                onsets = batch['onsets'].to(device).transpose(1, 2)

                # forward pass
                out = self(spec)
                loss = criterion(out, truth, onsets)

                # backward pass
                optim.zero_grad()
                loss.backward()
                optim.step()

                # calculate precision, recall and f1 score
                with torch.no_grad():
                    e10 = err(truth, out, onsets, 1e-2)
                    e3 = err(truth, out, onsets, 1e-3)
                    e1 = err(truth, out, onsets, 1e-4)
                
                # update epoch loss
                epoch_loss += loss.item()
                ee1 += e1
                ee3 += e3
                ee10 += e10

                if verbose:
                    data_iter.set_description(
                        f'Epoch {epoch + 1}/{epochs} - Loss: {1e3*epoch_loss/(i+1):.1f} - 10%: {100*ee10/(i+1):.2f}% - 3%: {100*ee3/(i+1):.2f}% - 1%: {100*ee1/(i+1):.2f}%')
                else:
                    print(
                        f'Epoch {epoch + 1}/{epochs} - Loss: {1e3*epoch_loss/(i+1):.1f}')

            # perform validation if validation data is not None
            if validation_data:
                result = self.val_split(
                    validation_data,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    device=device
                )

                print(f"Validation -  10%: {result[10]:.2f}% - 3%: {result[3]:.2f}% - 1%: {result[1]:.2f}%")

            # save model if save_path is not None, save model
            if save_path:
                torch.save(self.state_dict(), f"{save_path}_{train_start_time}.pt")
    

    def val_split(
        self,
        val_x,
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

        ee1, ee3, ee10 = 0, 0, 0

        self.eval()

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                # get data
                spec = batch['real'].to(device).transpose(1, 2)
                spec = (spec + 40)/40

                truth = batch[self.split].to(device).transpose(1, 2)
                onsets = batch['onsets'].to(device).transpose(1, 2)

                # forward pass
                out = self(spec)

                # calculate precision, recall and f1 score
                with torch.no_grad():
                    ee10 += err(truth, out, onsets, 1e-2)
                    ee3 += err(truth, out, onsets, 1e-3)
                    ee1 += err(truth, out, onsets, 1e-4)

        return {
            10: 100*ee10/(i+1),
            3: 100*ee3/(i+1),
            1: 100*ee1/(i+1)
        }


class PretrainedFrames(SplitModule):
    def __init__(self, onset_model, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.0):
        super().__init__('frames')
        self.onset_model = onset_model
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model+88)
        encoder_layers = TransformerEncoderLayer(d_model+88, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = ConvStack(ntoken, d_model, bias=True)#nn.Embedding(ntoken, d_model)
        self.d_model = math.sqrt(d_model)
        self.decoder = nn.Linear(d_model+88, 88)
    
    def forward(self, x):
        with torch.no_grad():
            onsets = self.onset_model(x)
        
        x = self.encoder(x)
        x = torch.cat([x, onsets], dim=2) * self.d_model
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, torch.zeros(x.size(1),x.size(1),device='cuda',requires_grad=False))
        x = nn.ReLU()(x)
        return self.decoder(x)

class OnsetsFrames(SplitModule):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.0, model_path=None, bias1=False, bias2=False):
        super().__init__('frames')
        self.onset_model = TransformerModel(ntoken, d_model, nhead, d_hid, nlayers, dropout, bias=bias1)
        if model_path:
            self.onset_model.load_state_dict(torch.load(model_path))
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model+88)
        encoder_layers = TransformerEncoderLayer(d_model+88, 20, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = ConvStack(ntoken, d_model, bias=bias2)
        self.d_model = math.sqrt(d_model)
        self.decoder = nn.Linear(d_model+88, 88)
    
    def forward(self, x):
        onsets = self.onset_model(x)
        x = self.encoder(x)
        x = torch.cat([x, onsets], dim=2) * self.d_model
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, torch.zeros(x.size(1),x.size(1),device='cuda',requires_grad=False))
        x = nn.ReLU()(x)
        return onsets, self.decoder(x)

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
        train_start_time = int(time()) % 1000000
        
        if verbose:
            print(sum(p.numel()
                      for p in self.parameters() if p.requires_grad), 'parameters')

        data_loader = DataLoader(
            x, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        criterion = loss_fn()
        optim = optimizer(self.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss, epoch_onset_P, epoch_onset_R, epoch_onset_F1, epoch_frame_P, epoch_frame_R, epoch_frame_F1 = 0, 0, 0, 0, 0, 0, 0
            # start training model
            self.train()
            data_iter = tqdm(enumerate(data_loader), ascii=True, total=len(
                data_loader)) if verbose else enumerate(data_loader)
            for i, batch in data_iter:
                # get data
                spec = batch['real'].to(device).transpose(1, 2)
                spec = (spec + 40)/40

                onsets = batch['onsets'].to(device).transpose(1, 2)
                truth = batch['frames'].to(device).transpose(1, 2)

                # forward pass
                out_onsets, out_frames = self(spec)
                loss = criterion(out_onsets, onsets) + criterion(out_frames, truth)

                # backward pass
                optim.zero_grad()
                loss.backward()
                optim.step()

                # calculate precision, recall and f1 score
                onsets_pred = out_onsets > 0
                frames_pred = out_frames > 0
                onsets = onsets > 0
                truth = truth > 0
                with torch.no_grad():
                    onsets_P = precision(onsets, onsets_pred)
                    onsets_R = recall(onsets, onsets_pred)
                    frame_P = precision(truth, frames_pred)
                    frame_R = recall(truth, frames_pred)
                    # update loss, precision, recall, f1 score and min/max
                    epoch_loss += loss.item()
                    # epoch_P += P
                    # epoch_R += R
                    # epoch_F1 += 2 * P * R / (P + R + 1e-8)
                    epoch_onset_P += onsets_P
                    epoch_onset_R += onsets_R
                    epoch_onset_F1 += 2 * onsets_P * onsets_R / (
                        onsets_P + onsets_R + 1e-8)
                    epoch_frame_P += frame_P
                    epoch_frame_R += frame_R
                    epoch_frame_F1 += 2 * frame_P * frame_R / (
                        frame_P + frame_R + 1e-8)

                    if verbose:
                        data_iter.set_description(
                            f'Epoch {epoch + 1}/{epochs}:L={1e3*epoch_loss/(i+1):.2f},oP={100*epoch_onset_P/(i+1):.2f},oR={100*epoch_onset_R/(i+1):.2f},oF1={100*epoch_onset_F1/(i+1):.2f},fP={100*epoch_frame_P/(i+1):.2f},fR={100*epoch_frame_R/(i+1):.2f},fF1={100*epoch_frame_F1/(i+1):.2f}')

            # perform validation if validation data is not None
            if validation_data:
                result = self.val_split(
                    validation_data,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    device=device
                )

                if verbose:
                    print(
                        f"Validation - oP: {result['oP']:.2f}, oR: {result['oR']:.2f}, oF1: {result['oF1']:.2f}, fP: {result['fP']:.2f}, fR: {result['fR']:.2f}, fF1: {result['fF1']:.2f}")
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
        device='cuda',
        onset_thresh=0,
        frame_thresh=0
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

        onset_P, onset_R, onset_F1, frame_P, frame_R, frame_F1 = 0, 0, 0, 0, 0, 0

        self.eval()

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                # get data
                spec = batch['real'].to(device).transpose(1, 2)
                spec = (spec + 40)/40

                onsets = batch['onsets'].to(device).transpose(1, 2)
                truth = batch['frames'].to(device).transpose(1, 2)

                # forward pass
                out_onsets, out_frames = self(spec)

                # calculate precision, recall and f1 score
                onsets_pred = out_onsets > onset_thresh
                frames_pred = out_frames > frame_thresh
                onsets = onsets > 0
                truth = truth > 0
                
                oP = precision(onsets, onsets_pred)
                oR = recall(onsets, onsets_pred)
                fP = precision(truth, frames_pred)
                fR = recall(truth, frames_pred)


                # accumulate precision, recall
                onset_P += oP
                onset_R += oR
                frame_P += fP
                frame_R += fR
                onset_F1 += 2 * oP * oR / (oP + oR + 1e-8)
                frame_F1 += 2 * fP * fR / (fP + fR + 1e-8)

        return {
            'oP': 100 * onset_P /   (len(data_loader)),
            'oR': 100 * onset_R /   (len(data_loader)),
            'oF1': 100 * onset_F1 / (len(data_loader)),
            'fP': 100 * frame_P /   (len(data_loader)),
            'fR': 100 * frame_R /   (len(data_loader)),
            'fF1': 100 * frame_F1 / (len(data_loader))
        }

    
    

if __name__ == '__main__':
    dataset = OnsetsFramesVelocity(output_path)
    val_dataset = OnsetsFramesVelocity(output_path, split='val')
    test_set = OnsetsFramesVelocity(output_path, split='test')

    # # load velocity model
    # model = VelocityModel(229, 512, 16, 512, 8)
    # model.load_state_dict(torch.load("./velocity_190898.pt"))
    # model.cuda()
    # model.fit(dataset, validation_data=val_dataset, epochs=12, lr=4e-5, loss_fn=modified_mse, save_path='velocity')

    # load onsets frames model
    model = OnsetsFrames(229, 512, 16, 512, 8, bias2=True, dropout=0.1)
    model.load_state_dict(torch.load("./final8p_307822.pt"))
    model.cuda()

    # import numpy as np
    # x = np.array([-10,-5, -2, -1, -0.7, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 2, 5, 10])
    # y = []

    # for thresh in x:
    #     y.append(model.val_split(val_dataset, batch_size=32, onset_thresh=thresh, frame_thresh=thresh))
    # print(x)
    # print(y)

    # import matplotlib.pyplot as plt
    # plt.plot(x,y['oP'], label='oP')
    # plt.plot(x,y['oR'], label='oR')
    # plt.plot(x,y['oF1'], label='oF1')
    # plt.plot(x,y['fP'], label='fP')
    # plt.plot(x,y['fR'], label='fR')
    # plt.plot(x,y['fF1'], label='fF1')
    # plt.legend()
    # plt.show()

    model.fit(dataset, validation_data=val_dataset, epochs=6, lr=3e-6, save_path='final8p', batch_size=2)
