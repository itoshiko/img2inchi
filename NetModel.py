import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import time
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer, Linear)
from torch.nn import Dropout, LayerNorm, MultiheadAttention
from typing import Optional
from utils import PAD_ID

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PositionalEncodingNd(nn.Module):
    def __init__(self, d_pos: int, max_size: int, emb_size: int):
        super(PositionalEncodingNd, self).__init__()
        self.emb_size = emb_size
        self.d_pos = d_pos
        den = torch.exp(- torch.arange(0, emb_size, 2 * d_pos) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_size).reshape(max_size, 1)
        pos_embedding = torch.zeros((max_size, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        for dim in range(self.d_pos):  # dim == 0; 1
            prepad = dim * 2 * self.num  # 0; 256
            postpad = self.d_model - (dim + 1) * 2 * self.num  # 256; 0
            embed = self.pos_embedding[:, :x.shape[dim]]
            embed = F.pad(embed, (0, 0, prepad, postpad))  # [512, 14]
            shape = [1] * dim + embed.shape[0] + [1] * (self.d_pos - dim + 2) + embed.shape[1]
            embed.view(shape)
            token_embedding += embed  # [1, 512, 14, 1]; [1, 512, 1, 14]
        return token_embedding

class DecoderLayer(nn.Module):
    def __init__(self, dim_encoder, dim_decoder, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(dim_decoder, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(dim_decoder, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(dim_decoder, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, dim_decoder)
        self.memory_linear = Linear(dim_encoder, dim_decoder)

        self.norm1 = LayerNorm(dim_decoder)
        self.norm2 = LayerNorm(dim_decoder)
        self.norm3 = LayerNorm(dim_decoder)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(DecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        memory = self.memory_linear(memory) #dim_encoder -> dim_decoder
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Img2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, patch_size: int,
                 dim_encoder: int, dim_decoder: int, nhead: int, vocab_size: int,
                 dim_feedforward:int = 512, dropout:float = 0.1):
        super(Img2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=dim_encoder, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = DecoderLayer(dim_encoder=dim_encoder, dim_decoder=dim_decoder, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.patch_emb = Linear(patch_size * patch_size, dim_encoder)
        self.seq_emb = Linear(patch_size * patch_size, dim_encoder)
        self.generator = Linear(dim_decoder, vocab_size)
        self.positional_encoding_img = PositionalEncodingNd(d_pos=2, max_size=225, emb_size=dim_encoder)
        self.positional_encoding_seq = PositionalEncodingNd(d_pos=1, max_size=300, emb_size=dim_decoder)

    def forward(self, img: Tensor, seq: Tensor):
        memory = self.encode(img)
        outs = self.decode(seq, memory)
        return self.generator(outs)

    def img_split(self, img: Tensor):
        patch = []
        return patch

    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, img: Tensor):
        img_emb = self.positional_encoding_img(self.patch_emb(self.img_split(img)))
        return self.transformer_encoder(src=img_emb, src_mask=None, src_padding_mask=None)

    def decode(self, seq: Tensor, memory: Tensor):
        seq_emb = self.positional_encoding_seq(self.seq_emb(seq))
        tgt_mask = self.generate_square_subsequent_mask(seq.shape[0])
        tgt_padding_mask = (seq == PAD_ID).transpose(0, 1) if self.trainning else None
        return self.transformer_decoder(tgt=seq_emb, memory=memory, tgt_mask=tgt_mask,
                                        tgt_padding_mask=tgt_padding_mask, memory_key_padding_mask=None)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
