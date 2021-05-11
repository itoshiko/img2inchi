import math
import torch
import torch.nn as nn
from torch import Tensor

class one_hot():
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def encode(self, seqs):
        batch_size, seq_len = seqs.shape
        seqs = seqs.unsqueeze(-1)
        one_hot_en = torch.zeros(batch_size, seq_len, self.vocab_size)
        one_hot_en.scatter_(dim=2, index=seqs.long(), src=torch.ones(batch_size, seq_len, self.vocab_size))
        return one_hot_en

    def __call__(self, seqs):
        return self.encode(seqs)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
