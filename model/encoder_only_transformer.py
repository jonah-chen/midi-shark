from common import SplitModule, ConvStack, PositionalEncoding
import math
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from database import output_path, OnsetsFramesVelocity

class TransformerModel(SplitModule):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1, split='onsets'):
        super().__init__(split)
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = ConvStack(ntoken, d_model)#nn.Embedding(ntoken, d_model)
        self.d_model = math.sqrt(d_model)
        self.decoder = nn.Linear(d_model, 88)

    def forward(self, src):
        src = self.encoder(src) * self.d_model
        src = self.pos_encoder(src)
        src = self.transformer_encoder(src, torch.zeros(src.size(1),src.size(1),device='cuda',requires_grad=False))
        src = nn.ReLU()(src)
        return self.decoder(src)

if __name__ == '__main__':
    dataset = OnsetsFramesVelocity(output_path)
    val_dataset = OnsetsFramesVelocity(output_path, split='val')
    model = TransformerModel(229, 512, 16, 512, 8, split='frames')
    model.cuda()

    model.fit(dataset, validation_data=val_dataset, epochs=12, lr=8e-5)
