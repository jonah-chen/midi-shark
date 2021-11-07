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
from constants import *
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

EPOCHS = 5

class DeeplabV3(Module):
    def __init__(self):
        super(DeeplabV3, self).__init__()
        self.encoder = DeepLabv3Encoder()
        self.decoder = ImageDecoder(34)

    def forward(self, x):
        # checkpoint the model
        x = checkpoint(self.encoder, x, preserve_rng_state=False)
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

    loader = DataLoader(data, batch_size=16, shuffle=True, num_workers=24)

    model = DeeplabV3()
    model.cuda()
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), 
                    lr=0.045, 
                    momentum=0.9)
    print(sum(p.numel() for p in model.parameters()))

    for _ in range(EPOCHS):
        for _, (img, label) in tqdm(enumerate(loader), total=len(loader)):
            img,label = img.cuda(),label.cuda()
            out = model(img)
            loss = criterion(out, label)
            optimizer.zero_grad()
            # print(loss)
            loss.backward()
            optimizer.step()
            # print(loss.item())
            # del out, loss, img, label
            # torch.cuda.empty_cache()
