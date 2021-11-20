from database import SpectrogramNotesDataset, output_path
import sys
sys.path.append('../')

from deeplab import DeepLabv3Encoder, ImageDecoder
import torch
from torch.nn import BCEWithLogitsLoss, Module
from torch.optim import SGD
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import os
from random import randint
import numpy as np
from tqdm import tqdm

class DeNoiseModel(Module):
    def __init__(self):
        super(DeNoiseModel, self).__init__()
        self.encoder = DeepLabv3Encoder(1)
        self.decoder = ImageDecoder(1)

    def forward(self, x):
        # checkpoint the model
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.squeeze(1)

EPOCHS = 5

if __name__ == '__main__':
    data = SpectrogramNotesDataset(output_path)

    loader = DataLoader(data, batch_size=8, shuffle=True, num_workers=24, drop_last=True)

    model = DeNoiseModel()
    model.cuda()
    optimizer = SGD(model.parameters(), 
                    lr=0.005, 
                    momentum=0.9)
    print(sum(p.numel() for p in model.parameters()))

    for epoch in range(EPOCHS):
        # start summing up the loss over epoch
        # epoch_iou = 0
        epoch_loss = 0
        epoch_iou = 0
        iterable_loader = tqdm(enumerate(loader), total=len(loader), ascii=True)
        for i, sample in iterable_loader:
            img,label = sample['real'].cuda(), sample['notes'].cuda()
            tt = label.long()
            criterion = BCEWithLogitsLoss(pos_weight=torch.tensor([img.shape[1]*img.shape[2]/torch.sum(tt).item()], device='cuda:0'))
            
            # normalize image between 0 and 1
            img = (img + 80) / 80

            out = model(img)
            loss = criterion(out, label)
            optimizer.zero_grad()
            # print(loss)
            loss.backward()
            optimizer.step()
            
            # batch_iou = 0
            # for class_num in range(34):
            #     pp = pred_label == class_num
            #     tt = label == class_num
            #     batch_iou += torch.sum(pp & tt).item() / (torch.sum(pp | tt).item() + 1e-8)
            # batch_iou /= 34
            # print(batch_iou)
            pp = (out > 0.5)
            intersection = torch.sum(pp & tt).item()
            union = torch.sum(pp | tt).item()
            batch_iou = intersection / (union + 1e-8)
            epoch_iou += batch_iou
            
            epoch_loss += loss.item()

            # calculate some metrics
            diff = torch.square(out - label)

            iterable_loader.set_postfix_str(f'BCE={1e3*loss.item():.1f} avg:{1e3*epoch_loss/(i+1):.2f} i={100*intersection/229/862:.1f}% u={100*union/229/862:.1f}% iou={1e2*batch_iou:.1f}% avg:{1e2*epoch_iou/(i+1):.2f}%')

        # # perform validation on the validation set
        # val_iou = 0
        # val_loss = 0
        # with torch.no_grad():
        #     for _, (img, label) in enumerate(val_loader):
        #         img,label = img.cuda(),label.cuda()
        #         out = model(img)
        #         loss = criterion(out, label)
        #         pred_label = torch.argmax(out,axis=1)
                
        #         batch_iou = 0
        #         for class_num in range(34):
        #             pp = pred_label == class_num
        #             tt = label == class_num
        #             batch_iou += torch.sum(pp & tt).item() / (torch.sum(pp | tt).item() + 1e-8)
        #         batch_iou /= 34
        #         # print(batch_iou)
        #         val_iou += batch_iou
        #         val_loss += loss.item()

        # report epoch loss
        epoch_loss /= len(loader)
        epoch_iou /= len(loader)
        print(f'Epoch: {epoch+1} | Loss: {1e3*epoch_loss} | IoU: {1e2 * epoch_iou}')
        # # report validation loss
        # val_loss /= len(val_loader)
        # val_iou /= len(val_loader)
        # print('Validation Loss: {} | Validation IoU: {}'.format(val_loss, val_iou))
    
    # save the model to model_dir
    torch.save(model.state_dict(), 'modelspec.pth')
