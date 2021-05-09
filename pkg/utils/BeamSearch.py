import operator
from queue import PriorityQueue

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


class BeamSearchNode(object):
    def __init__(self, hiddenState, cellState, previousNode, wordId, logProb, length):
        """
        :param hiddenState:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        """
        self.h = hiddenState
        self.c = cellState
        self.prevNode = previousNode
        self.wordid = wordId
        self.log_p = logProb
        self.l = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.log_p / float(self.l - 1 + 1e-6) + alpha * reward


def beam_decode(decoder, encodings, beam_width=10, top_k=1):
    """
    :param decoder: decoder of network that implements decoder_step
    :param encodings: encoder output
    :param seqs:
    :param beam_width: width of beam search
    :param top_k: how many sentence do you want to generate
    :return: decoded_batch
    """

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
    # Initialize LSTM state
    h, c = decoder.init_hidden_state(encodings)  # (batch_size, dim_decoder)

    # decoding goes sentence by sentence
    for idx in range(batch_size):
        # Start with the start of the sentence token
        decoder_input = torch.ones(1, 1).fill_(SOS_ID).type(torch.long).to(device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((top_k + 1), top_k - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(h, c, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000:
                break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h
            decoder_cell = n.c

            if n.wordid.item() == EOS_ID and n.prevNode is not None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_hidden, decoder_cell, decoder_output = decoder.decode_step(encodings, decoder_hidden,
                                                                               decoder_cell, decoder_input)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width, dim=1)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, decoder_cell, n, decoded_t, n.log_p + log_p, n.l + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(top_k)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = [n.wordid]
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch


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
