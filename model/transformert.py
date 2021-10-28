import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformer import PositionalEmbedding
import copy

class FeedForward(nn.Module):
    def __init__(self, dim_model = 512, dim_ff=2048):
        super().__init__() 
        self.linear1 = nn.Linear(dim_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, dim_model)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class Residualdropout(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        x = self.sublayer(*tensors)
        #print(self.sublayer(*tensors))
        #print(torch.is_tensor(tensors[-1]))
        return self.norm(tensors[-1] + self.dropout(x[0]))

class EncoderLayer(nn.Module):
    #insert dim_k,dim_v
    def __init__(
        self,
        dim_model = 512, 
        num_heads = 8, 
        dim_ff = 2048, 
        dropout = 0.1,
    ):
        super().__init__()
        multihead = torch.nn.MultiheadAttention
        self.attention = Residualdropout(
            multihead(dim_model, num_heads, dropout, batch_first=True),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residualdropout(
            FeedForward(dim_model, dim_ff),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.attention(x, x, x)
        return self.feed_forward(x)

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(
        self,
        dim_in, 
        num_layers = 6,
        dim_model = 512, 
        num_heads = 8, 
        dim_ff = 2048, 
        dropout = 0.1, 
    ):
        super().__init__()
        self.embedding = nn.Embedding(dim_in,dim_model)
        self.pe = PositionalEmbedding()
        self.layers = get_clones(EncoderLayer(dim_model, num_heads, dim_ff, dropout), num_layers)

    def forward(self, x: Tensor, num_layers) -> Tensor:
        #x = self.embedding(x)
        x = self.pe(x)
        for i in range(num_layers):
            x = self.layers[i](x)
        return x


src = torch.rand(64, 16, 512)
src = torch.tensor(src)
print(src.shape)
tEncoder = Encoder(
            dim_in = 3,
            num_layers = 6,
            dim_model = 512, 
            num_heads = 8, 
            dim_ff = 2048, 
            dropout = 0.1, 
        )
out = tEncoder(src,6)
print(out.shape)
