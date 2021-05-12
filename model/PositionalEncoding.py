import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class PositionalEncodingNd(nn.Module):
    def __init__(self, d_pos: int, max_size: int, d_model: int):
        """
        Embedding the (absolute) positional encodings to some data

        :param d_pos: the dimension of positional space
        :param max_size: the max lenth of all positional dimension
        :param d_model: the dimension of features at every position, or the dimension of the model
        """
        super(PositionalEncodingNd, self).__init__()
        self.d_model = d_model
        self.d_pos = d_pos
        den = torch.exp(- torch.arange(0, d_model, 2 * d_pos) * math.log(10000) / d_model)
        self.num = len(den)
        pos = torch.arange(0, max_size).reshape(max_size, 1)
        # pos_embedding: shape(max_size, d_model)
        pos_embedding = torch.zeros(max_size, 2 * self.num)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x: Tensor):
        """
        :param token_embedding: shape (batch_size, (positional), d_model)
        :return: the data after embedding positional encodings
        """
        input_shape = x.shape
        for dim in range(self.d_pos):  # dim == 0; 1
            prepad = dim * 2 * self.num  # 0; 256
            postpad = self.d_model - (dim + 1) * 2 * self.num  # 256; 0
            embed = self.pos_embedding[:input_shape[dim + 1], :]
            embed = F.pad(embed, (prepad, postpad, 0, 0))  # [512, 14]
            shape = [1] * (dim + 1) + [embed.shape[0]] + [1] * (self.d_pos - dim - 1) + [self.d_model]
            embed = embed.view(shape)
            x += embed  # [1, 512, 14, 1]; [1, 512, 1, 14]
        return x
