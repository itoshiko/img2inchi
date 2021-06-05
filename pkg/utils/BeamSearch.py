import heapq

from torch.nn.functional import softmax
from pkg.utils.vocab import EOS, SOS
from typing import Any, Callable, Iterable, Optional, TypeVar, Union

import torch
import numpy as np
from torch import Tensor, LongTensor
from pkg.utils.utils import flatten_list, split_list

PAD_ID: int = 0
SOS_ID: int = 1
EOS_ID: int = 2

class BeamSearchNode(object):
    """
    Node for beam search. It stores the infomation included:
    current decode memory, current sequence, sequence probability, batch_id
    """

    def __init__(self, decode_memory, wordId: int, prob: float, parent=None, batch_id: Optional[int]=None):
        """
        :param decode_memory: the memory data while decoding. Used as input in decode_step
        :param wordId: new word gives by its id
        :param prob: the conditional probability that takes this word
        :param parent: the previous node that this node was generated from
        :param batch_id: the batch that this sequence comes from
        """
        self.batch_id = batch_id
        self.decode_memory = decode_memory
        if parent is not None:
            pre_seq = parent.seq
            pre_prob = parent.prob
            self.batch_id = parent.batch_id
        else:
            pre_seq = []
            pre_prob = 1
        self.seq: list[int] = pre_seq + [wordId]
        self.prob: float = pre_prob * prob
        if self.batch_id is None: raise Exception('Cannot get the batch_id')

    def __lt__(self, other):
        return self.prob < other.prob

T = TypeVar('T')
def _delete_by_index(list1: 'list[T]', index: 'list[int]') -> 'list[T]':
    index = list(set(range(len(list1))) - set(index))
    return [list1[i] for i in index]

class BeamSearch(object):
    """
    Base class for beam search. It implements method for beam search: beam_decode.
    But while beam search, we need to initialize the decode memory and excute the decode step, 
    which is left to be implemented. Please inherit this class and implement them.
    Then you can use beam_decode method for beam search decoding.
    """

    def __init__(self, device: str='cpu', beam_width: int=10, topk: int=1, max_len: int=200, max_batch: int=1):
        """
        :param decoder: decoder of network that will be used to decode.
        """
        self.beam_width = beam_width
        topk = min(topk, beam_width)
        self.topk = topk
        self.max_len = max_len
        self.max_batch = max_batch
        self.device = device

    def init_beam(self, encode_memory: Tensor):
        batch_size, _, d_model = encode_memory.shape
        beam_width = self.beam_width

        inputs: Tensor = torch.ones((batch_size, ), dtype=torch.int16, device=self.device) * SOS_ID

        decode_memory = self.init_decode_memory(encode_memory)

        logits = self.decode_step(decode_memory=decode_memory, inputs=inputs)

        probas = softmax(logits, dim=-1)
        probas, inputs = torch.topk(probas, beam_width, dim=-1)
        probas = probas.view(-1)
        inputs = inputs.view(-1)
        self.trans_decode_memory(
            decode_memory, 
            (
                torch.ones((batch_size, beam_width), device=self.device) * 
                torch.arange(batch_size, device=self.device).unsqueeze(-1)
            ).view(-1).long()
        )
        return decode_memory, probas, inputs

    def beam_decode(self, encode_memory: Tensor) -> Tensor:
        """
        Run beam search and decode the encode_memory (with no gradient).

        :param encode_memory: encodings, encoder output
        :param beam_width: width of beam search
        :return: a list of beam_decode answers. Each element is a list of the topk decoded seq of each batch.
        The outer list is arranged by batch order, the inner list is arranged by decending order of scores.
        """
        batch_size = encode_memory.shape[0]
        beam_width = self.beam_width
        seqs: Tensor = torch.zeros((batch_size * beam_width, self.max_len + 1), dtype=torch.int16, device=self.device)
        total_probas: Tensor = torch.ones((batch_size * beam_width, 1), dtype=torch.float32, device=self.device)
        finished: Tensor = torch.zeros((batch_size * beam_width, ), dtype=torch.bool, device=self.device)
        base_ind = torch.arange(0, batch_size, device=self.device).unsqueeze(-1)

        decode_memory, total_probas[:, 0], inputs = self.init_beam(encode_memory)
        seqs[:, 0] = SOS_ID
        seqs[:, 1] = inputs

        # decode
        for t in range(2, self.max_len + 1):
            logits = self.decode_step(decode_memory=decode_memory, inputs=inputs)
            probas = softmax(logits, dim=-1)
            probas, indices1 = torch.topk(probas, beam_width, dim=-1, sorted=False)
            probas = (probas * total_probas).view(batch_size, beam_width ** 2)
            probas, indices2 = torch.topk(probas, beam_width, dim=-1, sorted=False)
            total_probas[:, 0] = probas.view(-1)
            parent_indices = (torch.floor(indices2 / beam_width) + base_ind * beam_width).view(-1).long()
            self.trans_decode_memory(decode_memory, parent_indices)
            seqs = seqs[parent_indices]
            finished = finished[parent_indices]
            chosen_indices = (indices2 + base_ind * (beam_width ** 2)).view(-1).long()
            inputs = indices1.view(-1)[chosen_indices]
            inputs[finished] = PAD_ID
            seqs[:, t] = inputs
            finished = finished | (inputs == EOS_ID)
            if finished.all():
                break
        seqs = seqs[:, :(t + 1)]
        i = torch.argmax(seqs[:, -1])
        total_probas = total_probas.view(batch_size, beam_width)
        _, indices = torch.topk(total_probas, self.topk, dim=-1)
        indices = (indices + base_ind * beam_width).view(-1).long()
        seqs = seqs[indices]
        return seqs

    def decode_with_width_one(
            self, encode_memory: Tensor, choose_func: Callable[[Tensor], 'LongTensor']
        ) -> Tensor:
        if encode_memory.ndim == 2:
            encode_memory = encode_memory.unsqueeze(0)
        batch_size = encode_memory.shape[0]
        inputs: Tensor = torch.ones((batch_size, ), dtype=int, device=self.device) * SOS_ID
        seqs: Tensor = torch.ones((batch_size, self.max_len), dtype=int, device=self.device) * SOS_ID
        not_end: Tensor = torch.ones((batch_size, ), dtype=int, device=self.device).unsqueeze(-1)

        # Initialize decode_memory
        # Start with the start of the sentence token for each seq
        decode_memory = self.init_decode_memory(encode_memory=encode_memory)

        # decode
        for t in range(self.max_len):
            logits = self.decode_step(decode_memory=decode_memory, inputs=inputs)
            inputs = choose_func(logits)
            seqs[:, t] = inputs
            not_end[:, 0] = not_end[:, 0] * (inputs != EOS_ID)
            if (1 - not_end).all().item():
                break
        return seqs[:, :(t + 1)]

    def greedy_decode(self, encode_memory: Tensor) -> Tensor:
        def greedy_func(logits: Tensor):
            return torch.max(softmax(logits.squeeze(1), dim=-1), dim=-1)[-1]
        return self.decode_with_width_one(encode_memory=encode_memory, choose_func=greedy_func)

    def sample_decode(self, encode_memory: Tensor) -> Tensor:
        def sample_func(logits):
            indexes = torch.multinomial(softmax(logits.squeeze(1), dim=-1), num_samples=1, replacement=True).squeeze(-1)
            return indexes
        return self.decode_with_width_one(encode_memory=encode_memory, choose_func=sample_func)

    def sample(self, encode_memory: Tensor, gts: Tensor, forcing_num: int, vocab_size: int) -> 'tuple[Tensor, list[Tensor]]':
        if encode_memory.ndim == 2:
            encode_memory = encode_memory.unsqueeze(0)
        batch_size = encode_memory.shape[0]
        N = gts.shape[1]
        inputs: list[int] = [SOS_ID for _ in range(batch_size)]
        seqs: list[list[int]] = [[SOS_ID] for _ in range(batch_size)]

        # Initialize decode_memory
        # Start with the start of the sentence token for each seq
        decode_memory = self.init_decode_memory(encode_memory=encode_memory)
        logits = torch.zeros((batch_size, N, vocab_size), device=self.device)

        # decode
        for t in range(N):
            logits[:, t, :] = self.decode_step(decode_memory_list=decode_memory, inputs=inputs)
            if t < forcing_num:
                for k in range(batch_size):
                    inputs[k] = int(gts[k, t].item())
            else:
                probs = softmax(logits[:, t, :], dim=-1)
                indexes = torch.multinomial(probs, num_samples=1, replacement=True)
                for k in range(batch_size):
                    inputs[k] = int(indexes[k].item())
            for k in range(batch_size):
                seqs[k].append(inputs[k])

        sampled = [torch.tensor(seq, dtype=torch.int, device=self.device) for seq in seqs]
        return logits, sampled

    def init_decode_memory(self, encode_memory: Tensor) -> list:
        '''
        Initialize the decode_memory by encode_memory. This method should implemented by subclass.

        :param encode_memory: encode_memory, or encodings
        :return: a list of decode memory for each batch
        '''
        raise NotImplementedError("Method 'init_decode_memory' isn't implemented.")

    def decode_step(self, decode_memory: list, inputs: Tensor) -> Tensor:
        '''
        Decode for single step. This method should implemented by subclass.

        :param decode_memory_list: a list of decode memory for decode_step.
        This method will directly modify this parameter after decoding.

        :param inputs: a list of word_id for this step as input.

        :return: the logits after decoding.
        '''
        raise NotImplementedError("Method 'decode_step' isn't implemented.")

    def trans_decode_memory(self, decode_memory: list, indices: Tensor) -> None:
        raise NotImplementedError("Method 'decode_step' isn't implemented.")


'''
def greedy_decode(decoder, encodings, seqs, sample_decode):
    dim_encoder = encodings.shape[-1]
    if encodings.ndim == 3:  # single input, add batch=1
        encodings = encodings.unsqueeze(0).view(1, -1, dim_encoder)
    elif encodings.ndim == 4:  # batch mode
        batch_size = encodings.shape[0]
        encodings = encodings.view(batch_size, -1, dim_encoder)
    else:
        raise NotImplementedError("Unknown num of dims: {}".format(encodings.ndim))
    batch_size = encodings.shape[0]
    decoder_hidden, decoder_cell = decoder.init_hidden_state(encodings)  # (batch_size, dim_decoder)
    decode_lengths = [len(seq) - 1 for seq in seqs]
    decoded_batch = torch.zeros((batch_size, max(decode_lengths)))
    decoder_input = torch.LongTensor([[SOS_ID] for _ in range(batch_size)], device=device)

    for t in range(max(decode_lengths)):
        decoder_hidden, decoder_cell, decoder_output = decoder.decode_step(encodings, decoder_hidden,
                                                                           decoder_cell, decoder_input)
        if sample_decode:
            sample_i = torch.multinomial(torch.softmax(decoder_output), 1, True)
            sample_i = sample_i.view(-1)
            decoded_batch[:, t] = sample_i
            decoder_input = sample_i.detach().view(-1, 1)
        else:
            top_v, top_i = decoder_output.data.topk(1, dim=1)  # get candidates
            top_i = top_i.view(-1)
            decoded_batch[:, t] = top_i
            decoder_input = top_i.detach().view(-1, 1)

    return decoded_batch
'''