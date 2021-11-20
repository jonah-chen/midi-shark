import sys
sys.path.append('../')

from deeplab import DeepLabv3Encoder, ImageDecoder
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import SGD
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
import os
from random import randint
import numpy as np
from tqdm import tqdm


class Cityscapes(datasets.Cityscapes):
    t = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop((512,512), scale=(0.9,1.0))       
    ])

    def __getitem__(self, i):
        inp, target = super(Cityscapes, self).__getitem__(i)
        seed = randint(0, 0xffffffffffffffff)
        
        torch.manual_seed(seed)
        inp = Cityscapes.t(inp)
        torch.manual_seed(seed)
        target = Cityscapes.t(target)
        return inp, torch.Tensor(np.asarray(target)).long()

EPOCHS = 100

class DeeplabV3(Module):
    def __init__(self):
        super(DeeplabV3, self).__init__()
        self.encoder = DeepLabv3Encoder(3)
        self.decoder = ImageDecoder(34, out_size=(512,512))

    def forward(self, x):
        # checkpoint the model
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    # if torch.cuda.is_available():  # Use GPU if and only if available
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #     torch.set_default_dtype(torch.float32)

    data = Cityscapes('/media/hina/LinuxStorage/Datasets/cityscapes', 
                     target_type='semantic', 
                     transform=transforms.ToTensor())
    val_data = Cityscapes('/media/hina/LinuxStorage/Datasets/cityscapes',
                            target_type='semantic',
                            split='val',
                            transform=transforms.ToTensor())

    loader = DataLoader(data, batch_size=4, shuffle=True, num_workers=24)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=True, num_workers=24)

    model = DeeplabV3()
    model.cuda()
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), 
                    lr=0.09, 
                    momentum=0.9)
    print(sum(p.numel() for p in model.parameters()))

    for epoch in range(EPOCHS):
        # start summing up the loss over epoch
        epoch_iou = 0
        epoch_loss = 0
        iterable_loader = tqdm(enumerate(loader), total=len(loader))
        for i, (img, label) in iterable_loader:
            img,label = img.cuda(),label.cuda()
            img.requires_grad = True
            out = model(img)
            loss = criterion(out, label)
            optimizer.zero_grad()
            # print(loss)
            loss.backward()
            optimizer.step()

            pred_label = torch.argmax(out,axis=1)
            
            batch_iou = 0
            for class_num in range(34):
                pp = pred_label == class_num
                tt = label == class_num
                batch_iou += torch.sum(pp & tt).item() / (torch.sum(pp | tt).item() + 1e-8)
            batch_iou /= 34
            # print(batch_iou)
            epoch_iou += batch_iou
            epoch_loss += loss.item()
            iterable_loader.set_postfix(ordered_dict={'IoUm': epoch_iou/(i+1), 'IoU': batch_iou, 'loss': loss.item()})

        # perform validation on the validation set
        val_iou = 0
        val_loss = 0
        with torch.no_grad():
            for _, (img, label) in enumerate(val_loader):
                img,label = img.cuda(),label.cuda()
                out = model(img)
                loss = criterion(out, label)
                pred_label = torch.argmax(out,axis=1)
                
                batch_iou = 0
                for class_num in range(34):
                    pp = pred_label == class_num
                    tt = label == class_num
                    batch_iou += torch.sum(pp & tt).item() / (torch.sum(pp | tt).item() + 1e-8)
                batch_iou /= 34
                # print(batch_iou)
                val_iou += batch_iou
                val_loss += loss.item()

        # report epoch loss
        epoch_loss /= len(loader)
        epoch_iou /= len(loader)
        print('Epoch: {} | Loss: {} | IoU: {}'.format(epoch+1, epoch_loss, epoch_iou))
        # report validation loss
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        print('Validation Loss: {} | Validation IoU: {}'.format(val_loss, val_iou))
    
    # save the model to model_dir
    torch.save(model.state_dict(), 'model.pth')