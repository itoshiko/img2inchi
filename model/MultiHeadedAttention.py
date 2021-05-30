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
                padding_mask: Optional[Tensor]=None, dropout: Optional[nn.Module]=None):
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
    p_attn: Tensor = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
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
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.displaying = False
        self.attn = None
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, 
                pos_mask: Optional[Tensor]=None, padding_mask: Optional[Tensor]=None):
        '''
        Implements Figure 2
        :param query, key, value: after linear layer
        :param pos_mask: mask the position that should ignore. using bool type and True means mask/ignore, 
        float type and 0.0 means mask/ignore
        :param padding_mask: the padding position in key that should be masked/ignored
        '''
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = [
            x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for x in (query, key, value)
        ]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, pos_mask=pos_mask, 
                        padding_mask=padding_mask, dropout=self.dropout)
        if not self.displaying:
            self.attn = None

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linear(x)

    def display(self, displaying=True):
        if not displaying:
            self.attn = None
        self.displaying = displaying
    
    def get_attention(self):
        if not self.displaying:
            raise Exception('Not display mode! Cannot get the attention matrix.')
        return self.attn


class multiLinear(nn.Module):
    def __init__(self, d_model, num=3):
        super(multiLinear, self).__init__()
        self.linears = clones(nn.Linear(d_model, d_model), num)

    def forward(self, *inputs: Tensor) -> 'list[Tensor]':
        return [l(x) for l, x in zip(self.linears, inputs)]
