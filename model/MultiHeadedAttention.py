import copy
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query: Tensor, key: Tensor, value: Tensor, pos_mask: Optional[Tensor]=None, 
                padding_mask: Optional[Tensor]=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if pos_mask is not None:
        if pos_mask.dtype == torch.bool:
            scores = scores.masked_fill(pos_mask, float('-inf'))
        else:
            scores = scores.masked_fill(pos_mask == 0, float('-inf'))
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(-2)
        if padding_mask.dtype == torch.bool:
            scores = scores.masked_fill(padding_mask, float('-inf'))
        else:
            scores = scores.masked_fill(padding_mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim = -1)
    return torch.matmul(p_attn, value)

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, nhead):
        '''
        Take in model size and number of heads.
        nhead must be divisible by model size.
        batch_first is True.
        '''
        super(MultiHeadedAttention, self).__init__()
        assert d_model % nhead == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // nhead
        self.h = nhead
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, 
                pos_mask: Optional[Tensor]=None, padding_mask: Optional[Tensor]=None):
        "Implements Figure 2"
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x = attention(query, key, value, pos_mask=pos_mask, 
                                padding_mask=padding_mask)
        del query, key, value

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
