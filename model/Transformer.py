from model.Extractor import FeaturesExtractor
from typing import Optional, NewType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear

from model.MultiHeadedAttention import MultiHeadedAttention, multiLinear, clones
from model.PositionalEncoding import PositionalEncodingNd
from model.TokenEmbedding import TokenEmbedding


PAD_ID = 0
pretrained = "model weights"

class EncoderLayer(nn.Module):
    '''

    '''
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=2048, dropout: float=0.1, activation: str="relu"):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu(inplace=True)
        super(EncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, pos_mask=src_mask, padding_mask=src_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model

        self.self_attn = MultiHeadedAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.src_attn = MultiHeadedAttention(d_model=d_model, nhead=nhead, dropout=dropout)

        # (tgt, tgt, tgt)
        self.self_attn_linears = multiLinear(d_model=d_model, num=3)
        # (tgt, memory, memory)
        self.src_attn_linears = multiLinear(d_model=d_model, num=3)
        self.self_attn_memory = None
        self.src_attn_memory = None

        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout, inplace=True)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu(inplace=True)
        super(DecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, 
                tgt_mask: Optional[Tensor] = None, tgt_paddingg_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None, memory_padding_mask: Optional[Tensor] = None,
                decode_mem: Optional[list] = None) -> Tensor:
        '''
        :param tgt: target. shape: (barch_size, max_len, d_model)
        '''
        if decode_mem is not None:
            memk1, memv1, memk2, memv2 = decode_mem

        q, k, v = self.self_attn_linears(tgt, tgt, tgt)
        if decode_mem is not None:
            k, v = [torch.cat((x, y), dim=1) for x, y in zip([memk1, memv1], [k, v])]
            self.self_attn_memory = [k, v]
        tgt2 = self.self_attn(q, k, v, pos_mask=tgt_mask, padding_mask=tgt_paddingg_mask)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        q, k, v = self.src_attn_linears(tgt, memory, memory)
        if decode_mem is not None:
            k, v = [torch.cat((x, y), dim=1) for x, y in zip([memk2, memv2], [k, v])]
            self.src_attn_memory = [k, v]
        tgt2 = self.src_attn(q, k, v, pos_mask=memory_mask, padding_mask=memory_padding_mask)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def clear_cache(self):
        self.self_attn_memory = None
        self.src_attn_memory = None

    def init_decode_memory(self, memory: Tensor, device) -> 'list[Tensor]':
        batch_size = memory.shape[0]
        return [init_empty_tensor(batch_size=batch_size, d_model=self.d_model, device=device), 
                init_empty_tensor(batch_size=batch_size, d_model=self.d_model, device=device)] + \
                self.src_attn_linears(
                    init_empty_tensor(batch_size=batch_size, d_model=self.d_model, device=device), 
                    memory, memory
                )[1:]

    def display(self, displaying):
        self.src_attn.display(displaying)
    
    def get_attention(self) -> Tensor:
        return self.src_attn.get_attention()

class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer: EncoderLayer, num_layers: int, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=src_mask, src_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer: DecoderLayer, num_layers: int, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, 
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, 
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                decode_mem_list: Optional[list] = None) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        for i in range(self.num_layers):
            mod = self.layers[i]
            decode_mem = decode_mem_list[i] if decode_mem_list is not None else None
            output = mod(
                output, memory, 
                tgt_mask=tgt_mask, tgt_paddingg_mask=tgt_key_padding_mask,
                memory_mask=memory_mask, memory_padding_mask=memory_key_padding_mask,
                decode_mem=decode_mem
            )
            if decode_mem_list is not None:
                decode_mem_list[i] = mod.self_attn_memory + mod.src_attn_memory
        
        if self.norm is not None:
            output = self.norm(output)

        return output
    
    def init_decode_memory(self, memory: Tensor, device) -> 'list[list[Tensor]]':
        decode_mem_list = [
            self.layers[i].init_decode_memory(memory=memory, device=device)
            for i in range(self.num_layers)
        ]
        return decode_mem_list

    def display(self, displaying):
        for mod in self.layers:
            mod.display(displaying)

    def get_attention(self) -> 'list[Tensor]':
        return [mod.get_attention() for mod in self.layers]

    def clear_cache(self):
        for i in range(self.num_layers):
            self.layers[i].clear_cache()


class Img2SeqTransformer(nn.Module):
    def __init__(self, feature_size: 'tuple[int, int]', extractor_name: str, pretrain: str,
                max_seq_len: int, tr_extractor: bool, num_encoder_layers: int, 
                num_decoder_layers: int, d_model: int, nhead: int, vocab_size: int,
                dim_feedforward: int = 1024, dropout: float = 0.1):
        super(Img2SeqTransformer, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.d_model = d_model
        self.features_extractor = FeaturesExtractor(num_features=d_model, output_size=feature_size, 
                                                extractor_name=extractor_name, pretrain=pretrain, tr_extractor=tr_extractor)
        encoder_layer = EncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = DecoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = Linear(d_model, vocab_size)
        self.seq_emb = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding_seq = PositionalEncodingNd(d_pos=1, max_size=max_seq_len, d_model=d_model, dropout=dropout)

    def forward(self, img: Tensor, seq: Tensor):
        memory = self.encode(img)
        return self.decode(seq, memory)

    def scst(self, run=True):
        if run:
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.transformer_encoder.parameters():
                param.requires_grad = True

    def encode(self, img: Tensor, ft_size: 'Optional[list[int, int]]'=None) -> Tensor:
        '''
        batch_size = img.shape[0]
        '''
        features = self.features_extractor(img)
        batch_size,  ft_h, ft_w, n_feature = features.shape
        if ft_size is not None:
            ft_size[0] = ft_h
            ft_size[1] = ft_w
        features = features.view(batch_size, -1, n_feature)
        return self.transformer_encoder(src=features, src_mask=None, src_key_padding_mask=None)

    def decode(self, seq: Tensor, memory: Tensor) -> Tensor:
        seq_emb = self.positional_encoding_seq(self.seq_emb(seq))
        tgt_mask = self.generate_square_subsequent_mask(seq.shape[1], seq.device)
        tgt_padding_mask = seq == PAD_ID
        outs = self.transformer_decoder(tgt=seq_emb, memory=memory, tgt_mask=tgt_mask, memory_mask=None,
                                        tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=None)
        return self.generator(outs)

    def init_decode_memory(self, memory: Tensor):
        self.clear_cache()
        return self.transformer_decoder.init_decode_memory(memory=memory, device=self.device)

    def decode_step(self, seq: Tensor, decode_memory: list, pos: Optional[int]=None, tgt_padding_mask: Tensor=None) -> Tensor:
        '''
        decode for single step
        :param seq: the new input words. shape: (batch_size, 1)
        :param decode_mem_list: the memory while decoding. It stores the [key, value] for previous words.
        '''
        if seq.ndim == 1:
            seq = seq.unsqueeze(-1)
        if pos is None:
            pos = decode_memory[0][0].shape[1]
        if tgt_padding_mask is not None:
            assert tgt_padding_mask.shape[1] == pos + 1
        batch_size = seq.shape[0]
        memory = init_empty_tensor(batch_size=batch_size, d_model=self.d_model, device=self.device)
        seq_emb = self.positional_encoding_seq(x=self.seq_emb(seq), pos=(pos,))
        outs = self.transformer_decoder(tgt=seq_emb, memory=memory, tgt_mask=None, memory_mask=None,
                                        tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=None, 
                                        decode_mem_list=decode_memory)
        return self.generator(outs).squeeze(1)

    def clear_cache(self):
        self.transformer_decoder.clear_cache()

    def display(self, displaying=True):
        self.transformer_decoder.display(displaying=displaying)

    def get_attention(self):
        return self.transformer_decoder.get_attention()

    def generate_square_subsequent_mask(self, size: int, device: str):
        mask = 1.0 - torch.triu(torch.ones((size, size), device=device), 1)
        # mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

TransformerType = NewType('TransformerType', Img2SeqTransformer)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def init_empty_tensor(batch_size: int, d_model: int, device):
    return torch.zeros((batch_size, 0, d_model), device=device)

'''
def _cat(input: 'tuple[Tensor]', dim: int=0):
    input = tuple([x for x in input if x is not None])
    return torch.cat(input, dim=dim)
'''