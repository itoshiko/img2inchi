import time
import math

import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getWH(img_w, img_h):
    img_w, img_h = np.ceil(img_w / 2), np.ceil(img_h / 2)
    img_w, img_h = np.ceil(img_w / 2), np.ceil(img_h / 2)
    img_w, img_h = np.ceil(img_w / 2), np.ceil(img_h / 2)
    img_w, img_h = np.ceil(img_w - 2), np.ceil(img_h - 2)
    return int(img_w), int(img_h)


class EncoderCNN(nn.Module):
    def __init__(self, img_w, img_h):
        super(EncoderCNN, self).__init__()
        self.cnn = nn.Sequential(
            # conv + max pool -> /2
            # 64 个 3*3 filters, strike = (1, 1), output_img.shape = ceil(L/S) = ceil(input/strike) = (H, W)
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv + max pool -> /2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # regular conv -> id
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            # conv
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        w, h = getWH(img_w, img_h)
        self.positionalEncoding = PositionalEncodingNd(2, max(w, h), 512)

    def forward(self, img):
        """
        Args:
            img: [batch, channel, W, H]
        return:
            out: [batch, W/2/2/2-2, H/2/2/2-2, 512]
        """
        out = self.positionalEncoding(self.cnn(img))
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, num_channel)
        batch_size, _, _, dim_encoder = out.shape
        # Flatten encoded_image
        out = out.view(batch_size, -1, dim_encoder)  # (batch_size, num_pixels, dim_encoder)
        return out


class PositionalEncodingNd(nn.Module):
    def __init__(self, d_pos: int, max_size: int, d_model: int):
        """
        Embedding the (absolute) positional encodings to some data

        :param d_pos: the dimension of positional space
        :param max_size: the max lenth of each positional dimension
        :param d_model: the dimension of features at every position, or the dimension of the model
        """
        self.d_model = d_model
        self.d_pos = d_pos
        den = torch.exp(- torch.arange(0, d_model, 2 * d_pos) * math.log(10000) / d_model).unsqueeze(1)
        pos = torch.arange(0, max_size).unsqueeze(0)
        self.num = den.shape[0]
        pos_embedding = torch.zeros((max_size, self.num))
        pos_embedding[0::2, :] = torch.sin(den * pos)  # even indices
        pos_embedding[1::2, :] = torch.cos(den * pos)  # odd  indices
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x: Tensor):
        """
        :param x: shape: (batch_size, d_model, (positional))
        :return: the data after embedding positional encodings
        """
        for dim in range(self.d_pos):  # dim == 0; 1
            prepad = dim * 2 * self.num  # 0; 256
            postpad = self.d_model - (dim + 1) * 2 * self.num  # 256; 0
            embed = self.pos_embedding[:, :x.shape[dim]]
            embed = F.pad(embed, (0, 0, prepad, postpad))  # [512, 14]
            shape = [1] + embed.shape[0] + [1] * dim + embed.shape[1] + [1] * (self.d_pos - dim + 1)
            embed.view(shape)
            x += embed  # [1, 512, 14, 1]; [1, 512, 1, 14]
        return x


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, dim_encoder, dim_decoder, dim_attention):
        """
        :param dim_encoder: feature size of encoded images
        :param dim_decoder: size of decoder's LSTM
        :param dim_attention: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(dim_encoder, dim_attention)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(dim_decoder, dim_attention)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(dim_attention, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encodings, hidden):
        """
        Forward propagation.

        :param encodings: encoded images, a tensor of shape (batch_size, num_pixels, dim_encoder)
        :param hidden: previous decoder output, a tensor of shape (batch_size, dim_decoder)
        :return: attention weighted encodings
        """
        att1 = self.encoder_att(encodings)  # (batch_size, num_pixels, dim_attention)
        att2 = self.decoder_att(hidden)  # (batch_size, dim_attention)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encodings = (encodings * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, dim_encoder)
        return attention_weighted_encodings


class DecoderWithAttention(nn.Module):
    """
    Decoder LSTM with Attention.
    """

    def __init__(self, dim_encoder, dim_decoder, dim_attention, dim_embed, vocab_size, dropout=0.5):
        """
        :param dim_encoder: feature size of encoded images
        :param dim_decoder: size of decoder's LSTM
        :param dim_attention: size of attention network
        :param dim_embed: dimension of embeded token
        :param vocab_size: size of vocabulary
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.dim_encoder = dim_encoder
        self.dim_decoder = dim_decoder
        self.dim_attention = dim_attention
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(dim_encoder, dim_decoder, dim_attention)  # attention network

        self.dropout = nn.Dropout(p=self.dropout)
        self.lstm = nn.LSTMCell(dim_embed + dim_encoder, dim_decoder, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(dim_encoder, dim_decoder)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(dim_encoder, dim_decoder)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(dim_decoder, dim_encoder)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.generator = nn.Linear(dim_decoder, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.generator.bias.data.fill_(0)
        self.generator.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_states(self, encodings):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encodings: encoded images, a tensor of shape (batch_size, num_pixels, dim_encoder)
        :return: hidden state, memory cell state
        """
        mean_encodings = encodings.mean(dim=1)
        h = self.init_h(mean_encodings)  # (batch_size, dim_decoder)
        c = self.init_c(mean_encodings)
        return h, c

    def decode_step(self, encodings, hidden, cell, seqs):
        """
        Running a single time-step of decoding.
        :param encodings: encoded images, a tensor of shape (batch_size, num_pixels, dim_encoder)
        :param hidden: hidden state
        :param cell: memory cell state
        :return: hidden, cell, preds (the predicted probability or scores of the next token)
        """
        attention_weighted_encodings = self.attention(encodings, hidden)
        gate = self.sigmoid(self.f_beta(hidden))  # gating scalar, (batch_size_t, dim_encoder)
        attention_weighted_encodings = gate * attention_weighted_encodings
        hidden, cell = self.lstm(torch.cat([seqs, attention_weighted_encodings], dim=1),
                                 (hidden, cell))  # (batch_size_t, dim_decoder)
        preds = self.generator(self.dropout(hidden))  # (batch_size_t, vocab_size)
        return hidden, cell, preds

    def forward(self, encodings, seqs):
        """
        Forward propagation.

        :param encodings: encoded images, a tensor of shape (batch_size, num_pixels, dim_encoder)
        :param seqs: encoded seqs in the decending order by sequence lenths, 
                     a tensor of shape (batch_size, max_sequence_length, dim_embed)
        :return: scores for vocabulary
        """

        batch_size = encodings.size(0)
        dim_encoder = self.dim_encoder
        vocab_size = self.vocab_size

        # Flatten image
        encodings = encodings.view(batch_size, -1, dim_encoder)  # (batch_size, num_pixels, dim_encoder)

        '''
        # Sort input data by decreasing lengths; why? apparent below
        seq_lengths, sort_ind = seq_lengths.squeeze(1).sort(dim=0, descending=True)
        encodings = encodings[sort_ind]
        seqs = seqs[sort_ind]
        '''

        # Initialize LSTM state
        h, c = self.init_hidden_state(encodings)  # (batch_size, dim_decoder)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = [len(seq) - 1 for seq in seqs]

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h, c, preds = self.decode_step(encodings, h[:batch_size_t], c[:batch_size_t],
                                           seqs[:batch_size_t, t, :])
            predictions[:batch_size_t, t, :] = preds

        return predictions


class Img2Seq(nn.Module):
    def __init__(self, img_w, img_h, vocab_size, dim_encoder, dim_decoder, dim_attention, dim_embed, dropout=0.5):
        """
        Encoder-Attention-Decoder
        Encoder: CNN
        Decoder: LSTM
        :param img_w: width of image
        :param img_h: height of image
        :param vocab_size: size of vocabulary
        :param dim_encoder: feature size of encoded images
        :param dim_decoder: size of decoder's LSTM
        :param dim_attention: size of attention network
        :param dim_embed: dimension of embeded token
        :param dropout: dropout
        """
        self.encoder = EncoderCNN(img_w, img_h)
        self.decoder = DecoderWithAttention(dim_attention, dim_embed, dim_decoder, dim_encoder, vocab_size, dropout)

    def encode(self, img):
        """
        Encode a (batch of) image(s) ith CNN.
        Return the encodings.
        :param img: Shape: (batch_size, w, h) or (w, h)
        """
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        return self.encoder(img)

    def init_hidden_states(self, encodings):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encodings.

        :param encodings: encoded images, a tensor of shape (batch_size, num_pixels, dim_encoder)
        :return: hidden state, memory cell state
        """
        return self.decoder.init_hidden_states(encodings)

    def decode_step(self, encodings, hidden, cell, seqs):
        """
        Running a single time-step of decoding.
        :param encodings: encoded images, a tensor of shape (batch_size, num_pixels, dim_encoder)
        :param hidden: hidden state
        :param cell: memory cell state
        :param seqs:
        :return: hidden, cell, preds (the predicted probability or scores of the next token)
        """
        return self.decoder.decode_step(encodings, hidden, cell, seqs)

    def forward(self, img, seqs):
        """
        Forward propagation.
        """
        encodings = self.encoder(img)
        return self.decoder(encodings, seqs)
