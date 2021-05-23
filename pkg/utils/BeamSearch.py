import heapq

from torch.nn.functional import softmax
from pkg.utils.vocab import SOS
from typing import Any, Callable, Optional, TypeVar

import torch
import numpy as np
from torch import Tensor, LongTensor
from pkg.utils.utils import flatten_list

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

    def beam_decode(self, encode_memory: Tensor) -> 'list[list[Tensor]]':
        """
        Run beam search and decode the encode_memory (with no gradient).

        :param encode_memory: encodings, encoder output
        :param beam_width: width of beam search
        :return: a list of beam_decode answers. Each element is a list of the topk decoded seq of each batch.
        The outer list is arranged by batch order, the inner list is arranged by decending order of scores.
        """
        batch_size = encode_memory.shape[0]
        decoded_batch: list[list[Tensor]] = []
        

        nodes: list[list[BeamSearchNode]] = []  # store all the active nodes for each seq in batch. 2d list
        end_nodes: list[list[BeamSearchNode]] = [[] * batch_size]   # store all the ended nodes for each seq. 2d list

        # Initialize decode_memory
        # Start with the start of the sentence token for each seq
        init_memory = self.init_decode_memory(encode_memory=encode_memory)
        nodes = [
            [BeamSearchNode(decode_memory=decode_memory, wordId=SOS_ID, prob=1, parent=None, batch_id=k)]
            for k, decode_memory in enumerate(init_memory)
        ]

        # decode
        for _ in range(self.max_len):
            # expand the nodes and get the successor
            nodes = self.expand_nodes(nodes)

            # check the ended nodes
            for k in range(batch_size):
                end_id = []
                for j in range(len(nodes[k])):
                    node = nodes[k][j]
                    if node.seq[-1] == EOS_ID:              # if the node has ended
                        end_id.append(j)                    # store its id
                        heapq.heappush(end_nodes[k], node)  # push the ended node
                        if len(end_nodes[k]) > self.topk:        # if end_nodes is too many
                            heapq.heappop(end_nodes[k])     # then pop the least scored node

                # delete all ended node from active nodes
                nodes[k] = _delete_by_index(nodes[k], end_id)
            if all([len(_nodes) == 0 for _nodes in nodes]):
                break

        # process the decoded seqs
        for k in range(batch_size):
            num_rem =  self.topk - len(end_nodes[k])
            if num_rem > 0:
                end_nodes[k] += nodes[k][:num_rem]
            decode_seq = [
                torch.tensor(node.seq, dtype=torch.int, device=self.device)
                for node in end_nodes[k]
            ]
            decoded_batch.append(decode_seq)
    
        return decoded_batch

    def decode_with_width_one(
            self, encode_memory: Tensor, choose_func: Callable[[Tensor, 'list[BeamSearchNode]'], 'tuple[Tensor, LongTensor]']
        ) -> 'list[Tensor]':
        if encode_memory.ndim == 2:
            encode_memory = encode_memory.unsqueeze(0)
        batch_size = encode_memory.shape[0]
        seqs: list[list[int]] = [[0] * batch_size]
        # Initialize decode_memory
        # Start with the start of the sentence token for each seq
        init_memory = self.init_decode_memory(encode_memory=encode_memory)
        nodes = [
            BeamSearchNode(decode_memory=decode_memory, wordId=SOS_ID, prob=1, parent=None, batch_id=k)
            for k, decode_memory in enumerate(init_memory)
        ]
        
        # decode
        for t in range(self.max_len):
            decode_memory_list, inputs = list(zip(*[self.read_node(node) for node in nodes]))
            decode_memory_list = list(decode_memory_list)
            inputs = list(inputs)
            logits = self.decode_step(decode_memory_list=decode_memory_list, inputs=inputs)
            probs, indexes = choose_func(logits, nodes)
            del_ind = []
            for k, node in enumerate(nodes):
                wordId = int(indexes[k].item())
                prob = float(probs[k].item())
                nodes[k] = BeamSearchNode(decode_memory=decode_memory_list[k], wordId=wordId, prob=prob, parent=node)
                if wordId == EOS_ID:
                    del_ind.append(k)
                    seqs[node.batch_id] = node.seq
            nodes = _delete_by_index(nodes, del_ind)
            if len(nodes) == 0:
                break
        return [torch.tensor(seq, dtype=torch.int, device=self.device) for seq in seqs]

    def greedy_decode(self, encode_memory: Tensor) -> 'list[Tensor]':
        def greedy_func(logits, nodes):
            probs = softmax(logits, dim=-1)
            return torch.max(probs, dim=-1)
        return self.decode_with_width_one(encode_memory=encode_memory, choose_func=greedy_func)

    def sample_decode(self, encode_memory: Tensor, gts: Tensor, forcing_num: int) -> 'tuple[Tensor, list[Tensor]]':
        global logits_list, i, end
        logits_list = []
        i = 0
        def _sample_func(logits, nodes):
            global logits_list, i, end
            logits_list.append(logits)
            if i < forcing_num:
                res = gts[:, i]
                end = res != PAD_ID
                res = res[end]
            else:
                res = torch.multinomial(softmax(logits), num_samples=1, replacement=True)
            i += 1
            return res
        result = self.decode_with_width_one(encode_memory=encode_memory, choose_func=_sample_func)
        logits = torch.cat(logits_list, dim=0)
        del logits_list, i, end
        return logits, result

    def read_node(self, node: BeamSearchNode) -> 'tuple[Any, int]':
        '''
        return the tuple: (decode_memory, wordId)
        '''
        return node.decode_memory, node.seq[-1]

    def expand_nodes(self, nodes: 'list[list[BeamSearchNode]]') -> 'list[list[BeamSearchNode]]':
        """
        Expand nodes. Do not call this method directly.

        :param beam_width: width of beam search
        :param decode_step: a function that decoder a single step
        :param encodings: encoder output
        :return: the successor
        """
        max_batch = self.max_batch
        batch_size = len(nodes)
        next_nodes: list[list[BeamSearchNode]] = [[] * max_batch]

        nodes = flatten_list(nodes)   # flatten
        
        for i in range(0, len(nodes), max_batch):
            batch_nodes = nodes[i:i + max_batch]
            decode_memory_list, inputs = list(zip(*[self.read_node(node) for node in batch_nodes]))
            decode_memory_list = list(decode_memory_list)
            inputs = list(inputs)
            logits = self.decode_step(decode_memory_list=decode_memory_list, inputs=inputs)
            probs = softmax(logits, dim=-1)
            probs, indexes = torch.topk(probs, self.beam_width, dim=-1)
            for k, decode_memory in enumerate(decode_memory_list):
                next_nodes[batch_nodes[k].batch_id] += [
                    BeamSearchNode(decode_memory=decode_memory, wordId=int(indexes[k, 0, j].item()), 
                                    prob=float(probs[k, 0, j].item()), parent=batch_nodes[k])
                    for j in range(self.beam_width)
                ]

        for k in range(batch_size):
            next_nodes[k] = heapq.nlargest(self.beam_width, next_nodes[k])

        return next_nodes

    def init_decode_memory(self, encode_memory: Tensor) -> list:
        '''
        Initialize the decode_memory by encode_memory. This method should implemented by subclass.

        :param encode_memory: encode_memory, or encodings
        :return: a list of decode memory for each batch
        '''
        raise NotImplementedError("Method 'init_decode_memory' isn't implemented.")

    def decode_step(self, decode_memory_list: list, inputs: 'list[int]') -> Tensor:
        '''
        Decode for single step. This method should implemented by subclass.

        :param decode_memory_list: a list of decode memory for decode_step.
        This method will directly modify this parameter after decoding.

        :param inputs: a list of word_id for this step as input.

        :return: the logits after decoding.
        '''
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