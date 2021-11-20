from database import DeNoiseDataset, output_path
import sys
sys.path.append('../')

from deeplab import DeepLabv3Encoder, ImageDecoder
import torch
from torch.nn import CrossEntropyLoss, Module, MSELoss
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

EPOCHS = 20

if __name__ == '__main__':
    data = DeNoiseDataset(output_path)

    loader = DataLoader(data, batch_size=8, shuffle=True, num_workers=24, drop_last=True)

    model = DeNoiseModel()
    model.cuda()
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), 
                    lr=0.004, 
                    momentum=0.9)
    print(sum(p.numel() for p in model.parameters()))

    for epoch in range(EPOCHS):
        # start summing up the loss over epoch
        # epoch_iou = 0
        epoch_loss = 0
        total_10 = 0
        total_5 = 0
        total_1 = 0
        iterable_loader = tqdm(enumerate(loader), total=len(loader), ascii=True)
        for i, sample in iterable_loader:
            img,label = sample['real'].cuda(), sample['generated'].cuda()

            # normalize image and label between 0 and 1
            img = (img + 80) / 80
            label = (label + 80) / 80

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
            # epoch_iou += batch_iou
            epoch_loss += loss.item()

            # calculate some metrics
            diff = torch.square(out - label)
            pred_10 = torch.sum(diff < 1e-2).item() * 100 / (out.shape[0] * out.shape[1] * out.shape[2])
            pred_5 = torch.sum(diff < 1e-3).item() * 100 / (out.shape[0] * out.shape[1] * out.shape[2])
            pred_1 = torch.sum(diff < 1e-4).item() * 100 / (out.shape[0] * out.shape[1] * out.shape[2])

            iterable_loader.set_postfix_str(f'MSE={1e3*loss.item():.1f} avg:{1e3*epoch_loss/(i+1):.2f} 10%={pred_10:.1f} 3.2%={pred_5:.1f} 1%={pred_1:.1f}')

            total_10 += pred_10
            total_5 += pred_5
            total_1 += pred_1

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
        total_10 /= len(loader)
        total_5 /= len(loader)
        total_1 /= len(loader)
        # epoch_iou /= len(loader)
        print(f'Epoch: {epoch+1} | Loss: {1e3*epoch_loss} | 10% : {total_10} | 3.2% : {total_5} | 1% : {total_1}')
        # # report validation loss
        # val_loss /= len(val_loader)
        # val_iou /= len(val_loader)
        # print('Validation Loss: {} | Validation IoU: {}'.format(val_loss, val_iou))
    
    # save the model to model_dir
    torch.save(model.state_dict(), 'model1.pth')