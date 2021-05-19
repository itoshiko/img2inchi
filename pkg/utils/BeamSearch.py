import heapq
from typing import Optional, Tuple

import torch
from torch.tensor import Tensor

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        if parent:
            pre_seq = parent.seq
            pre_prob = parent.prob
            self.batch_id = parent.batch_id
        else:
            pre_seq = []
            pre_prob = 1
        self.seq = pre_seq + [wordId]
        self.prob = pre_prob * prob
        if not batch_id: raise Exception('Cannot get the batch_id')

    def __lt__(self, other):
        return self.prob < other.prob


def _delete_by_index(list1, index):
    index = list(set(range(len(list1))) - set(index))
    return [list1[i] for i in index]

class BeamSearch(object):
    """
    Base class for beam search. It implements method for beam search: beam_decode.
    But while beam search, we need to initialize the decode memory and excute the decode step, 
    which is left to be implemented. Please inherit this class and implement them.
    Then you can use beam_decode method for beam search decoding.
    """

    def __init__(self, decoder, beam_width: int=10, topk: int=1, max_len: int=200, max_batch: int=1):
        """
        :param decoder: decoder of network that will be used to decode.
        """
        self.decoder = decoder
        self.beam_width = beam_width
        topk = min(topk, beam_width)
        self.topk = topk
        self.max_len = max_len
        self.max_batch = max_batch

    def beam_decode(self, encode_memory: Tensor) -> 'list[list[Tensor]]':
        """
        Run beam search and decode the encode_memory.

        :param encode_memory: encodings, encoder output
        :param beam_width: width of beam search
        :return: a list of beam_decode answers. Each element is a list of the topk decoded seq of each batch.
        The outer list is arranged by batch order, the inner list is arranged by decending order of scores.
        """
        dim_encoder = encode_memory.shape[-1]
        if encode_memory.ndim == 3:  # single input, add batch=1
            encode_memory = encode_memory.unsqueeze(0).view(1, -1, dim_encoder)
        elif encode_memory.ndim == 4:  # batch mode
            batch_size = encode_memory.shape[0]
            encode_memory = encode_memory.view(batch_size, -1, dim_encoder)
        else:
            raise NotImplementedError("Unknown num of dims: {}".format(encode_memory.ndim))
        batch_size = encode_memory.shape[0]
        decoded_batch = []
        

        nodes = []  # store all the active nodes for each seq in batch. 2d list
        end_nodes = [[] * batch_size]   # store all the ended nodes for each seq. 2d list

        # Initialize decode_memory
        # Start with the start of the sentence token for each seq
        for k, decode_memory in enumerate(self.init_decode_memory(encode_memory=encode_memory)):
            nodes.append([BeamSearchNode(decode_memory=decode_memory, wordId=SOS_ID, prob=1, parent=None, batch_id=k)])

        # decode
        for _ in range(self.max_len):
            # expand the nodes and get the successor
            nodes = self.expand_nodes(self.beam_width, encode_memory, nodes)

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

        # process the decoded seqs
        for k in range(batch_size):
            num_rem =  self.topk - end_nodes[k]
            if num_rem > 0:
                end_nodes[k] += nodes[k][:num_rem]
            end_nodes.sort(reverse=True)
            decode_seq = []
            for node in end_nodes[k]:
                decode_seq.append(Tensor(node.seq).long())
            decoded_batch.append(decode_seq)
    
        return decoded_batch

    def expand_nodes(self, nodes: 'list[BeamSearchNode]') -> 'list[BeamSearchNode]':
        """
        Expand nodes. Do not call this method directly.

        :param beam_width: width of beam search
        :param decode_step: a function that decoder a single step
        :param encodings: encoder output
        :return: the successor
        """
        # return the tuple: (decode_memory, wordId)
        read_node = lambda node: (node.decode_memory, node.seq[-1])
        max_batch = self.max_batch
        batch_size = len(nodes)
        next_nodes = [[] * max_batch]

        nodes = sum(nodes, [])   # flatten

        for i in range(0, len(nodes), max_batch):
            nodes_per_batch = nodes[i:i + max_batch]
            decode_memory_list, inputs = list(zip(*[read_node(node) for node in nodes_per_batch]))
            decode_answers = self.decode_step(decode_memory_list=decode_memory_list, inputs=inputs)
            for k, decode_ans_list in enumerate(decode_answers):
                next_nodes[nodes_per_batch[k].batch_id] += [
                    BeamSearchNode(*decode_ans, parent=nodes_per_batch[k])
                    for decode_ans in decode_ans_list
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

    def decode_step(self, decode_memory_list: list, inputs: 'list[int]') -> 'list[list[Tuple]]':
        '''
        Decode for single step. This method should implemented by subclass.

        :param decode_memory_list: a list of decode memory for decode_step.
        :param inputs: a list of word_id for this step.

        :return: list[list[tuple]]. Outer list corresponding to the decode answer for each input.
        Inner list gives the top beam_width answer. Each answer is a tuple:
        (new_decode_memory, new_word_id, conditional_probability)
        '''
        raise NotImplementedError("Method 'decode_step' isn't implemented.")


def greedy_decode(decoder, encodings, seqs):
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
        topv, topi = decoder_output.data.topk(1, dim=1)  # get candidates
        topi = topi.view(-1)
        decoded_batch[:, t] = topi

        decoder_input = topi.detach().view(-1, 1)

    return decoded_batch
