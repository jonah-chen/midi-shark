import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import copy
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class PositionalEmbedding(nn.Module):
    """
    A layer for positionally encoding tensors using the procedure
    described in section 3.5 of "Attention Is All You Need".
    """

    # A hashtable can be used to save the calculated sin/cos to avoid recomputation
    hashtable = dict()

    use_hashtable = False  # select whether to use a hashtable
    # defaults to False because it can consume a lot of memory

    @staticmethod
    def reset(val=dict()):
        PositionalEmbedding.hashtable = val

    def forward(self, x, start_pos=0):
        """ 
        The pytorch neural network forward function.

        Inputs:
            x: Input batch of sequences of tokens, each token is a vector. i.e. rank3 tensor
            [x.shape = (batch_sz, tokens, dims per token)]

        Outputs:
            Tensor y with the same shape as x, such that `y = x + PE` where
                `PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))`
                `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`
                meaning:
                    pos: position of the token
                    i:   the i-th element of the vector of the token at position pos
        Notes to self:
            Sin and Cos are expensive, power is expensive. 
            Adds/Multiplies are cheap. Division is bad. 
            Vectorize everything. Loops == BAD.
            Use log 10000.
        """
        batch_size, n_tokens, d_model = x.shape  # get input parameters

        if x.shape in PositionalEmbedding.hashtable:  # reuse the embeddings that are already generated
            return x + PositionalEmbedding.hashtable[(n_tokens, d_model)].repeat(batch_size, 1, 1)

        # half the dimension of model because there is both sine and cosine
        num_trigs = (d_model+1)//2
        # also account for odd and even cases

        _y = -math.log(10000) / d_model  # log table magic

        # Manual outer product is faster & require less intermediate variables than `torch.outer`
        _y = torch.exp(2*torch.arange(num_trigs, device=DEVICE)*_y).unsqueeze(0)
        _y = torch.arange(n_tokens, device=DEVICE).unsqueeze(1) * _y

        y = torch.empty((n_tokens, d_model), device=DEVICE)
        y[:, 0::2] = torch.sin(_y)
        y[:, 1::2] = torch.cos(_y[:, :num_trigs-(d_model % 2)])

        if PositionalEmbedding.use_hashtable:
            PositionalEmbedding.hashtable[(n_tokens, d_model)] = y
        return x + y.repeat(batch_size, 1, 1).to(DEVICE)

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