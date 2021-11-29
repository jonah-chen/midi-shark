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


class OnsetsBaseline(SplitModule):
    def __init__(self, input_features=229, output_features=88, model_complexity=48, split='onsets'):
        super().__init__(split)
        if split not in ['onsets', 'offsets', 'frames', 'velocities']:
            raise ValueError(
                'split must be either onsets, offsets, frames, or velocities')
        self.split = split
        model_size = model_complexity * 16

        self.conv = ConvStack(input_features, model_size)
        self.rnn = nn.LSTM(model_size, model_size//2,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(model_size, output_features)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.rnn(x)[0]
        x = nn.ReLU()(x)
        x = self.fc(x)
        return x
    
    

class OnsetsOffsetsFramesBaseline(nn.Module):
    def __init__(self, input_features=229, output_features=88, model_complexity=48):
        super().__init__()
        model_size = model_complexity * 16

        self.onsets_model = OnsetsBaseline(input_features, output_features, model_complexity)
        self.offsets_model = OnsetsBaseline(input_features, output_features, model_complexity)

        self.conv = ConvStack(input_features, model_size)
        self.rnn = nn.LSTM(model_size+output_features+output_features, model_size//2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(model_size, output_features)


    def forward(self, x):
        onsets = self.onsets_model(x)
        offsets = self.offsets_model(x)
        x = self.conv(x)
        x = torch.cat((x, onsets.detach(), offsets.detach()), dim=2)
        x = self.rnn(x)[0]
        x = nn.ReLU()(x)
        x = self.fc(x)
        return onsets, offsets, x
    
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
        train_start_time = int(time()) // 1000000

        if verbose:
            print(sum(p.numel()
                      for p in self.parameters() if p.requires_grad), 'parameters')

        data_loader = DataLoader(
            x, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        criterion = loss_fn()
        optim = optimizer(self.parameters(), lr=lr)

        for epoch in range(epochs):
            e_onsets_P, e_onsets_R, e_onsets_F1 = 0, 0, 0
            e_offsets_P, e_offsets_R, e_offsets_F1 = 0, 0, 0
            e_frames_P, e_frames_R, e_frames_F1 = 0, 0, 0
            epoch_loss = 0

            # start training model
            self.train()
            data_iter = tqdm(enumerate(data_loader), ascii=True, total=len(
                data_loader)) if verbose else enumerate(data_loader)
            for i, batch in data_iter:
                # get data
                spec = batch['real'].to(device).transpose(1, 2)
                spec = (spec + 40)/40

                r_onsets = batch['onsets'].to(device).transpose(1, 2)
                r_offsets = batch['offsets'].to(device).transpose(1, 2)
                r_frames = batch['frames'].to(device).transpose(1, 2)

                # forward pass
                out_onsets, out_offsets, out_frames = self(spec)
                loss_onsets = criterion(out_onsets, r_onsets)
                loss_offsets = criterion(out_offsets, r_offsets)
                loss_frames = criterion(out_frames, r_frames)

                loss = loss_onsets + loss_offsets + loss_frames
                epoch_loss += loss.item()

                # backward pass
                optim.zero_grad()
                loss.backward()
                optim.step()

                # calculate precision, recall and f1 score
                r_onsets = r_onsets > 0
                r_offsets = r_offsets > 0
                r_frames = r_frames > 0

                pred_onsets = out_onsets > 0
                pred_offsets = out_offsets > 0
                pred_frames = out_frames > 0

                with torch.no_grad():
                    P_onsets = precision(r_onsets, pred_onsets)
                    R_onsets = recall(r_onsets, pred_onsets)
                    F1_onsets = 2 * P_onsets * R_onsets / (
                        P_onsets + R_onsets + 1e-8)

                    P_offsets = precision(r_offsets, pred_offsets)
                    R_offsets = recall(r_offsets, pred_offsets)
                    F1_offsets = 2 * P_offsets * R_offsets / (
                        P_offsets + R_offsets + 1e-8)

                    P_frames = precision(r_frames, pred_frames)
                    R_frames = recall(r_frames, pred_frames)
                    F1_frames = 2 * P_frames * R_frames / (
                        P_frames + R_frames + 1e-8)

                # accumulate precision, recall, f1 score
                e_onsets_P += P_onsets
                e_onsets_R += R_onsets
                e_onsets_F1 += F1_onsets

                e_offsets_P += P_offsets
                e_offsets_R += R_offsets
                e_offsets_F1 += F1_offsets

                e_frames_P += P_frames
                e_frames_R += R_frames
                e_frames_F1 += F1_frames

                if verbose:
                    data_iter.set_description(
                        f'Epoch {epoch + 1}: Loss={1000*epoch_loss/(i+1):.1f}'+
                        f' (onsets)P={100*e_onsets_P/(i+1) :.2f},R={100*e_onsets_R/(i+1) :.2f},F1={100*e_onsets_F1/(i+1) :.2f}'+
                        f' (offsets)P={100*e_offsets_P/(i+1) :.2f},R={100*e_offsets_R/(i+1) :.2f},F1={100*e_offsets_F1/(i+1) :.2f}'+
                        f' (frames)P={100*e_frames_P/(i+1) :.2f},R={100*e_frames_R/(i+1) :.2f},F1={100*e_frames_F1/(i+1) :.2f}'
                    )
                else:
                    print(
                        f'Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss/(i+1):.4f}')

            # perform validation if validation data is not None
            if validation_data:
                result = self.val_oof(
                    validation_data,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    device=device
                )

                if verbose:
                    print(f"val(onsets) P: {100*result['onsets']['P'] :.2f} - R: {100*result['onsets']['R'] :.2f} - F1: {100*result['onsets']['F1'] :.2f}")
                    print(f"val(offsets) P: {100*result['offsets']['P'] :.2f} - R: {100*result['offsets']['R'] :.2f} - F1: {100*result['offsets']['F1'] :.2f}")
                    print(f"val(frames) P: {100*result['frames']['P'] :.2f} - R: {100*result['frames']['R'] :.2f} - F1: {100*result['frames']['F1'] :.2f}")
                else:
                    print(f"(onsets){100*result['onsets']['F1']:.2f} (offsets){100*result['offsets']['F1']:.2f} (frames){100*result['frames']['F1']:.2f}")

            # save model if save_path is not None, save model
            if save_path:
                torch.save(model.state_dict(), f"{save_path}_{train_start_time}.pt")
        
    
    def val_oof(
        self,
        val_x,
        batch_size=4,
        shuffle=True,
        num_workers=24,
        device='cuda'
    ):
        data_loader = DataLoader(
            val_x, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        onsets_P, onsets_R, onsets_F1 = 0, 0, 0
        offsets_P, offsets_R, offsets_F1 = 0, 0, 0
        frames_P, frames_R, frames_F1 = 0, 0, 0        

        self.eval()

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                # get data
                spec = batch['real'].to(device).transpose(1, 2)
                spec = (spec + 40)/40

                r_onsets = batch['onsets'].to(device).transpose(1, 2)
                r_offsets = batch['offsets'].to(device).transpose(1, 2)
                r_frames = batch['frames'].to(device).transpose(1, 2)
                
                # forward pass
                pred_onsets, pred_offsets, pred_frames = self(spec)
                
                # calculate precision, recall and f1 score
                r_onsets = r_onsets > 0
                r_offsets = r_offsets > 0
                r_frames = r_frames > 0

                pred_onsets = pred_onsets > 0
                pred_offsets = pred_offsets > 0
                pred_frames = pred_frames > 0

                # calculate precision, recall
                onsets_p = precision(r_onsets, pred_onsets)
                onsets_r = recall(r_onsets, pred_onsets)
                offsets_p = precision(r_offsets, pred_offsets)
                offsets_r = recall(r_offsets, pred_offsets)
                frames_p = precision(r_frames, pred_frames)
                frames_r = recall(r_frames, pred_frames)
                # accumulate precision, recall, f1 score, min, max
                onsets_P += onsets_p
                onsets_R += onsets_r
                onsets_F1 += 2 * onsets_p * onsets_r / (
                    onsets_p + onsets_r + 1e-8)

                offsets_P += offsets_p
                offsets_R += offsets_r
                offsets_F1 += 2 * offsets_p * offsets_r / (
                    offsets_p + offsets_r + 1e-8)

                frames_P += frames_p
                frames_R += frames_r
                frames_F1 += 2 * frames_p * frames_r / (
                    frames_p + frames_r + 1e-8)

        return {
            'onsets': {
                'P': onsets_P / len(data_loader),
                'R': onsets_R / len(data_loader),
                'F1': onsets_F1 / len(data_loader)
            },
            'offsets': {
                'P': offsets_P / len(data_loader),
                'R': offsets_R / len(data_loader),
                'F1': offsets_F1 / len(data_loader)
            },
            'frames': {
                'P': frames_P / len(data_loader),
                'R': frames_R / len(data_loader),
                'F1': frames_F1 / len(data_loader)
            }
        }


if __name__ == '__main__':
    dataset = OnsetsFramesVelocity(output_path)
    val_dataset = OnsetsFramesVelocity(output_path, split='val')

    model = OnsetsOffsetsFramesBaseline()
    model.cuda()
    model.fit(dataset)