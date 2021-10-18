import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    """
    Layer for positionally encoding sequences using the procedure
    described in section 3.5 of "Attention Is All You Need".

    `Each dimension of the positional encoding corresponds to a
    sinusoid. The wavelengths form a geometric progression from 2pi
    to 10000 * 2pi.`

    This, the dimension of the model is not as relevent as the 
    dimension of the embeddings. 

    Author: Jonah Chen
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

        Params:
            x: Input batch of sequences of tokens, each token is a vector.
               such that x.shape = (batch_sz, tokens, d_emb)

        Returns:
            y: Tensor with the same shape as x, such that `y = x + PE` where:
                
                `PE(pos, 2i)   = sin(pos / 10000^(2i/d_emb))`
                `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_emb))`
                
                with 
                pos: position of the token
                i:   the i-th element of the vector of the token at position pos)
        """
        batch_size, n_tokens, d_emb = x.shape  # get input parameters

        if x.shape in PositionalEmbedding.hashtable:  # reuse the embeddings that are already generated
            return x + PositionalEmbedding.hashtable[(n_tokens, d_emb)].repeat(batch_size, 1, 1)

        # half the dimension of input because there is both sine and cosine
        num_trigs = (d_emb+1)//2
        # also account for odd and even cases

        _y = -math.log(10000) / d_emb  # log table magic

        # Manual outer product is faster & require less intermediate variables than `torch.outer`
        _y = torch.exp(2*torch.arange(num_trigs)*_y).unsqueeze(0)
        _y = torch.arange(n_tokens).unsqueeze(1) * _y

        y = torch.empty((n_tokens, d_emb))
        y[:, 0::2] = torch.sin(_y)
        y[:, 1::2] = torch.cos(_y[:, :num_trigs-(d_emb % 2)])

        if PositionalEmbedding.use_hashtable:
            PositionalEmbedding.hashtable[(n_tokens, d_emb)] = y
        return x + y.repeat(batch_size, 1, 1)

if __name__ == '__main__':
    """
    Some [] tests for the various classes and methods

    Monitor GPU usage if 
    """
    from time import perf_counter, sleep  # Import timing control tools

    if torch.cuda.is_available():  # Use GPU if and only if available
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.set_default_dtype(torch.float32)

    print("Testing Creation of PositionalEmbedding Layer.")
    p = PositionalEmbedding()
    print("Created")

    print("\nTesting PositionalEmbeddings with zero matrices.")
    for d, e in [(3, 5), (4, 10), (7, 7), (10, 5)]:
        print(f"Case: {d}x{e}")
        print(p(torch.zeros((1, d, e,))))

    print("\nTesting PositionalEmbeddings with two matrices.")
    for d, e in [(3, 5), (4, 10), (7, 7), (10, 5)]:
        print(f"Case: {d}x{e}")
        print(p(2*torch.ones((1, d, e,))))

    print("\nTesting performance with random matrices.")
    for d in [1 << 10, 1 << 13, 1 << 16, 1 << 19]:
        total_time = 0
        mem, memr = 0, 0
        for _ in range(10):
            rmat = torch.randn((1, d, 512,))
            start = perf_counter()
            p(rmat)
            total_time += perf_counter() - start

            if torch.cuda.is_available():
                mem = max(mem, torch.cuda.memory_allocated())
                memr = max(memr, torch.cuda.memory_reserved())

        print(
            f"For size=({d},512,): Avg time={total_time*100:.2f}ms Mem={mem>>20}MB Reserved={memr>>20}MB")

    print("\nTesting batches")
    for batch_size in [4, 16, 64, 1024]:
        print(p(torch.randn((batch_size, 512, 512))).size())
        print(
            f"Mem={torch.cuda.memory_allocated()>>20}MB Reserved={torch.cuda.memory_reserved()>>20}MB")

    print("\nTesting hashtable.")
    PositionalEmbedding.use_hashtable = True
    for d in [1 << 10, 1 << 13, 1 << 16, 1 << 19]:
        total_time = 0
        mem, memr = 0, 0
        for _ in range(10):
            rmat = torch.randn((1, d, 512,))
            start = perf_counter()
            p(rmat)
            total_time += perf_counter() - start

            if torch.cuda.is_available():
                mem = max(mem, torch.cuda.memory_allocated())
                memr = max(memr, torch.cuda.memory_reserved())

        print(
            f"For size=({d},512,): Avg time={total_time*100:.2f}ms Mem={mem>>20}MB Reserved={memr>>20}MB")
    print(f"{len(PositionalEmbedding.hashtable)}. Expected 4.")

    print("\nTesting hashtable reset.")
    tmp_ht = PositionalEmbedding.hashtable
    PositionalEmbedding.reset()
    print(f"{len(PositionalEmbedding.hashtable)}. Expected 0.")
    PositionalEmbedding.reset(tmp_ht)
    print(PositionalEmbedding.hashtable.keys())
