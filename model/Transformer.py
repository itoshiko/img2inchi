import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, LayerNorm, Linear, Softmax
from torch.nn.modules import padding
from torchvision import models

from model.MultiHeadedAttention import MultiHeadedAttention, clones
from model.PositionalEncoding import PositionalEncodingNd
from model.TokenEmbedding import TokenEmbedding

PAD_ID = 0
pretrained = "model weights"



class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
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

    def __init__(self, decoder_layer: nn.Module, num_layers: int, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, 
                tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, 
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
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
        for mod in self.layers:
            output = mod(output, memory, 
                        tgt_mask=tgt_mask, tgt_paddingg_mask=tgt_key_padding_mask,
                        memory_mask=memory_mask, memory_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class EncoderLayer(nn.Module):
    '''

    '''
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=2048, dropout: float=0.1, activation: str="relu"):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(d_model, nhead)
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
        self.self_attn = MultiHeadedAttention(d_model, nhead)
        self.src_attn = MultiHeadedAttention(d_model, nhead)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
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
                memory_mask: Optional[Tensor] = None, memory_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        tgt2 = self.self_attn(tgt, tgt, tgt, pos_mask=tgt_mask, padding_mask=tgt_paddingg_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.src_attn(tgt, memory, memory, pos_mask=memory_mask, padding_mask=memory_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class FeaturesExtractor(nn.Module):
    def __init__(self, num_features: int=512, output_size: Tuple[int, int]=(16, 32), 
                extractor_name: str='resnet34', tr_extractor: bool=False):
        super(FeaturesExtractor, self).__init__()
        if extractor_name == 'resnet101':
            net = models.resnet101(pretrained=False)
            net.load_state_dict(torch.load(pretrained + '/ResNet101.pth'))
            dft_ft = 2048
        elif extractor_name == 'resnet34':
            net = models.resnet34(pretrained=False)
            net.load_state_dict(torch.load(pretrained + '/ResNet34.pth'))
            dft_ft = 512
        modules = list(net.children())[:-2]   # delete the last avgpool layer and fc layer.
        del net
        self.extractor = nn.Sequential(*modules)
        if not tr_extractor:
            for param in self.extractor.parameters():
                param.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=output_size) if output_size else None
        self.fc = Linear(dft_ft, num_features) if num_features != dft_ft else None

    def forward(self, img: Tensor):
        '''
        :param img: (batch_size, n_channel, H, W)
        :return: features. Shape: (output_w, output_h, batch_size, n_feature)
        '''
        if self.avgpool:
            ft = self.avgpool(self.extractor(img))      # (batch_size, n_feature, *output_size)
        else:
            ft = self.extractor(img)                    # (batch_size, n_feature, *default_size)
        ft = ft.permute(0, 2, 3, 1).contiguous()        # (batch_size, output_w, output_h, n_feature)
        return self.fc(ft) if self.fc else ft


class Img2SeqTransformer(nn.Module):
    def __init__(self, feature_size: Tuple[int, int], extractor_name: str, max_seq_len: int,
                tr_extractor: bool, num_encoder_layers: int, num_decoder_layers: int,
                d_model: int, nhead: int, vocab_size: int,
                dim_feedforward: int = 1024, dropout: float = 0.1):
        super(Img2SeqTransformer, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.d_model = d_model
        self.features_extractor = FeaturesExtractor(num_features=d_model, output_size=feature_size, 
                                                extractor_name=extractor_name, tr_extractor=tr_extractor)
        encoder_layer = EncoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = DecoderLayer(d_model=d_model, nhead=nhead,
                                        dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = Linear(d_model, vocab_size)
        self.seq_emb = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding_seq = PositionalEncodingNd(d_pos=1, max_size=max_seq_len, d_model=d_model)

    def forward(self, img: Tensor, seq: Tensor):
        memory = self.encode(img)
        outs = self.decode(seq, memory)
        return self.generator(outs)

    def encode(self, img: Tensor):
        '''
        batch_size = img.shape[0]
        '''
        features = self.features_extractor(img)
        batch_size,  _, _, n_feature = features.shape
        features = features.view(batch_size, -1, n_feature)
        return self.transformer_encoder(src=features, src_mask=None, src_key_padding_mask=None)

    def decode(self, seq: Tensor, memory: Tensor):
        seq_emb = self.positional_encoding_seq(self.seq_emb(seq))
        tgt_mask = self.generate_square_subsequent_mask(seq.shape[1], seq.device)
        tgt_padding_mask = seq == PAD_ID
        return self.transformer_decoder(tgt=seq_emb, memory=memory, tgt_mask=tgt_mask, memory_mask=None,
                                        tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=None)

    def generate_square_subsequent_mask(self, size: int, device: str):
        mask = 1.0 - torch.triu(torch.ones((size, size), device=device), 1)
        # mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
