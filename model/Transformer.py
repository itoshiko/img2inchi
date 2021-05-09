import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
from torch.nn import Dropout, LayerNorm, MultiheadAttention, Linear, Softmax
from typing import Optional
from torchvision import models

PAD_ID = 0
pretrained = "model weights"


class PositionalEncodingNd(nn.Module):
    def __init__(self, d_pos: int, max_size: int, d_model: int):
        """
        Embedding the (absolute) positional encodings to some data

        :param d_pos: the dimension of positional space
        :param max_size: the max lenth of each positional dimension
        :param d_model: the dimension of features at every position, or the dimension of the model
        """
        super(PositionalEncodingNd, self).__init__()
        self.d_model = d_model
        self.d_pos = d_pos
        den = torch.exp(- torch.arange(0, d_model, 2 * d_pos) * math.log(10000) / d_model)
        self.num = len(den)
        pos = torch.arange(0, max_size).reshape(max_size, 1)
        pos_embedding = torch.zeros(max_size, 2 * self.num)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        """
        :param token_embedding: shape ((positional), batch_size, d_model)
        :return: the data after embedding positional encodings
        """
        input_shape = token_embedding.shape
        for dim in range(self.d_pos):  # dim == 0; 1
            prepad = dim * 2 * self.num  # 0; 256
            postpad = self.d_model - (dim + 1) * 2 * self.num  # 256; 0
            embed = self.pos_embedding[:input_shape[dim], :]
            embed = F.pad(embed, (prepad, postpad, 0, 0))  # [512, 14]
            shape = [1] * dim + [embed.shape[0]] + [1] * (self.d_pos - dim) + [embed.shape[1]]
            embed = embed.view(shape)
            token_embedding += embed  # [1, 512, 14, 1]; [1, 512, 1, 14]
        return token_embedding

'''
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
'''

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class FeaturesExtractor(nn.Module):
    def __init__(self, num_features=512, output_size=(16, 32), extractor_name='resnet34', tr_extractor=False):
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

    def forward(self, img):
        '''
        :param img: (batch_size, n_channel, H, W)
        :return: features. Shape: (output_w, output_h, batch_size, n_feature)
        '''
        if self.avgpool:
            ft = self.avgpool(self.extractor(img))      # (batch_size, n_feature, *output_size)
        else:
            ft = self.extractor(img)                    # (batch_size, n_feature, *default_size)
        ft = ft.permute(2, 3, 0, 1)                     # (output_w, output_h, batch_size, n_feature)
        features = ft.contiguous()
        del ft
        return self.fc(features) if self.fc else features

class Img2SeqTransformer(nn.Module):
    def __init__(self, feature_size:(int, int), extractor_name: str, max_seq_len: int,
                tr_extractor: bool, num_encoder_layers: int, num_decoder_layers: int,
                d_model: int, nhead: int, vocab_size: int,
                dim_feedforward:int = 1024, dropout:float = 0.1):
        super(Img2SeqTransformer, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.d_model = d_model
        self.features_extractor = FeaturesExtractor(num_features=d_model, output_size=feature_size, 
                                                extractor_name=extractor_name, tr_extractor=tr_extractor)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.generator = Linear(d_model, vocab_size)
        self.seq_emb = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding_seq = PositionalEncodingNd(d_pos=1, max_size=max_seq_len, d_model=d_model)

    def forward(self, img: Tensor, seq: Tensor):
        seq = seq.transpose(0, 1).contiguous()
        memory = self.encode(img)
        outs = self.decode(seq, memory).transpose(0, 1).contiguous()
        return self.generator(outs)

    def encode(self, img: Tensor):
        '''
        batch_size = img.shape[0]
        '''
        features = self.features_extractor(img)
        _, _, batch_size, n_feature = features.shape
        features = features.view(-1, batch_size, n_feature)
        return self.transformer_encoder(src=features, mask=None, src_key_padding_mask=None)

    def decode(self, seq: Tensor, memory: Tensor):
        seq_emb = self.positional_encoding_seq(self.seq_emb(seq))
        tgt_mask = self.generate_square_subsequent_mask(seq.shape[0], seq.device)
        tgt_padding_mask = (seq == PAD_ID).transpose(0, 1)
        return self.transformer_decoder(tgt=seq_emb, memory=memory, tgt_mask=tgt_mask, memory_mask=None,
                                        tgt_key_padding_mask=tgt_padding_mask, memory_key_padding_mask=None)

    def generate_square_subsequent_mask(self, size, device):
        mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
