import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformer import PositionalEmbedding
import copy
import math

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
        return self.norm(tensors[0] + self.dropout(x[0]))

class EncoderLayer(nn.Module):
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
            multihead(dim_model, num_heads),
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
        #self.embedding = nn.Embedding(dim_in,dim_model)
        self.pe = PositionalEmbedding()
        self.layers = get_clones(EncoderLayer(dim_model, num_heads, dim_ff, dropout), num_layers)

    def forward(self, x: Tensor, num_layers) -> Tensor:
        #x = self.embedding(x)
        x = self.pe(x)
        for i in range(num_layers):
            x = self.layers[i](x)
        return x

class DecoderLayer(nn.Module):
    def __init__(
        self, 
        dim_model = 512, 
        num_heads = 8, 
        dim_ff = 2048, 
        dropout = 0.1, 
    ):
        super().__init__()
        multihead = torch.nn.MultiheadAttention
        self.attention1 = Residualdropout(
            multihead(dim_model, num_heads),
            dimension=dim_model,
            dropout=dropout,
        )
        self.attention2 = Residualdropout(
            multihead(dim_model, num_heads),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residualdropout(
            FeedForward(dim_model, dim_ff),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.attention1(tgt, tgt, tgt)
        tgt = self.attention2(tgt, memory, memory)
        return self.feed_forward(tgt)

class Decoder(nn.Module):
    def __init__(
        self, 
        num_layers = 6,
        dim_model = 512, 
        num_heads = 8, 
        dim_ff = 2048, 
        dropout = 0.1, 
    ):
        super().__init__()
        self.layers = get_clones(DecoderLayer(dim_model, num_heads, dim_ff, dropout), num_layers)
        self.linear = nn.Linear(dim_model, dim_model)
        self.pe = PositionalEmbedding()

    def forward(self, tgt: Tensor, memory: Tensor, num_layers) -> Tensor:
        tgt = self.pe(tgt)
        for i in range(num_layers):
            tgt = self.layers[i](tgt,memory)
        return torch.softmax(self.linear(tgt), dim =-1)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Transformer(nn.Module):
    def __init__(
        self, 
        emb_size: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        num_encoder_layers = 6,
        num_decoder_layers = 6,
        dim_model = 512, 
        num_heads = 8, 
        dim_ff = 512, 
        dropout: float = 0.1, 
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.encoder = Encoder(
            dim_in=3,
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_ff=dim_ff,
            dropout=dropout,
        )
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_ff=dim_ff,
            dropout=dropout,
        )
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        out = self.decoder(self.tgt_tok_emb(tgt), self.encoder(self.src_tok_emb(src),3),3)
        return self.generator(out)