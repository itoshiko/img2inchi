import heapq

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


class BeamSearchNode(object):
    def __init__(self, hiddenState, cellState, parent, wordId, prob, batch_id=None):
        """
        :param batch_id: the batch that this sequence comes from
        :param hiddenState, cellState:
        :param parent: the previous node that this node was generated from
        :param wordId: new word gives by its id
        :param prob: the conditional probability that takes this word
        """
        self.batch_id = batch_id
        self.h = hiddenState
        self.c = cellState
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


def delete_by_index(list1, index):
    index = list(set(range(len(list1))) - set(index))
    return [list1[i] for i in index]


def beam_decode(decoder, encodings, beam_width=10, topk=1, max_len=200):
    """
    :param decoder: decoder of network that implements decoder_step
    :param encodings: encoder output
    :param beam_width: width of beam search
    :return: decoded_batch
    """
    topk = min(topk, beam_width)
    dim_encoder = encodings.shape[-1]
    if encodings.ndim == 3:  # single input, add batch=1
        encodings = encodings.unsqueeze(0).view(1, -1, dim_encoder)
    elif encodings.ndim == 4:  # batch mode
        batch_size = encodings.shape[0]
        encodings = encodings.view(batch_size, -1, dim_encoder)
    else:
        raise NotImplementedError("Unknown num of dims: {}".format(encodings.ndim))
    batch_size = encodings.shape[0]
    decoded_batch = []
    decode_step = decoder.decode_step
    # Initialize LSTM state
    h, c = decoder.init_hidden_state(encodings)  # (batch_size, dim_decoder)

    nodes = []  # store all the active nodes for each seq in batch. 2d list
    end_nodes = [[] * batch_size]   # store all the ended nodes for each seq. 2d list

    # Start with the start of the sentence token for each seq
    for k in range(batch_size):
        nodes.append([BeamSearchNode(h[k, :], c[k, :], None, SOS_ID, 1, k)])

    # decode
    for _ in range(max_len):
        # expand the nodes and get the successor
        nodes = expand_nodes(beam_width, decode_step, encodings, nodes)

        # check the ended nodes
        for k in range(batch_size):
            end_id = []
            for j in range(len(nodes[k])):
                node = nodes[k][j]
                if node.seq[-1] == EOS_ID:              # if the node has ended
                    end_id.append(j)                    # store its id
                    heapq.heappush(end_nodes[k], node)  # push the ended node
                    if len(end_nodes[k]) > topk:        # if end_nodes is too many
                        heapq.heappop(node)             # then pop the least scored node

            # delete all ended node from active nodes
            nodes[k] = delete_by_index(nodes[k], end_id)

    # process the decoded seqs
    for k in range(batch_size):
        num_rem =  topk - end_nodes[k]
        if num_rem > 0:
            end_nodes[k] += nodes[k][:num_rem]
        decode_seq = []
        for node in end_nodes[k]:
            decode_seq.append(torch.Tensor(node.seq).long())
        decoded_batch.append(decode_seq)
    
    return decoded_batch


def expand_nodes(beam_width, decode_step, encodings, nodes):
    """
    :param beam_width: width of beam search
    :param decode_step: a function that decoder a single step
    :param encodings: encoder output
    :return: the successor
    """
    # return the tuple: (probability, encodings, wordId, hiddenState, cellState)
    read_node = lambda node: (node.prob, encodings[node.batch_id], node.seq[-1], node.h, node.c)
    batch_size = encodings.shape[0]
    next_nodes = [[] * batch_size]

    nodes = [node for node in _nodes for _nodes in nodes]   # flatten

    for i in range(0, len(nodes), batch_size):
        nodes_per_batch = nodes[i:i + batch_size]
        pre_probs, enc, inputs, h, c = list(zip(*[read_node(node) for node in nodes_per_batch]))
        inputs = torch.Tensor(inputs).long().to(device)
        enc = torch.stack(enc)
        h   = torch.stack(h)
        c   = torch.stack(c)

        # decode for one step using decode_step
        h, c, o = decode_step(encodings, h, c, inputs)

        probs = F.softmax(o, dim=1)
        probs, indexes = torch.topk(probs, beam_width, dim=1)
        for k in range(len(nodes_per_batch)):
            next_nodes[nodes_per_batch[k].batch_id] += [
                BeamSearchNode(h[k, :], c[k, :], nodes_per_batch[k], indexes[j], probs[k][j])
                for j in range(beam_width)
            ]

    for k in range(batch_size):
        next_nodes[k] = heapq.nlargest(beam_width, next_nodes[k])

    return next_nodes


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
