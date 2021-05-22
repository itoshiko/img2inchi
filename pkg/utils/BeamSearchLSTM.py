
import torch
from torch.nn.functional import softmax
from torch.nn import Module
from torch.tensor import Tensor

from pkg.utils.BeamSearch import BeamSearch
from model.Img2Seq import DecoderWithAttention

class BeamSearchLSTM(BeamSearch):
    """
    BeamSearch class for LSTM model.
    """

    def __init__(self, decoder: DecoderWithAttention, device: str='cpu', beam_width: int=10, topk: int=1, 
                max_len: int=200, max_batch: int=1):
        super(BeamSearchLSTM, self).__init__(device=device, beam_width=beam_width, topk=topk, 
                                                    max_len=max_len, max_batch=max_batch)
        self.decoder = decoder

    def init_decode_memory(self, encode_memory):
        '''
        Initialize the decode_memory by encode_memory using decoder.init_hidden_states.
        The decode memory is a tuple: (encode_memory/encoding, hidden state, memory cell)
        
        :param encode_memory: encode_memory, or encodings
        :return: a list of decode memory for each batch
        '''
        batch_size = encode_memory.shape[0]
        h, c = self.decoder.init_hidden_states(encode_memory)
        decode_memory_list = []
        for k in range(batch_size):
            decode_memory_list.append((encode_memory[k], h[k], c[k]))
        return decode_memory_list

    def decode_step(self, decode_memory_list: 'list[tuple]', inputs: 'list[int]') -> Tensor:
        '''
        Decode for single step using decoder.decode_step.

        :param decode_memory_list: a list of decode memory for decode_step.
        The decode memory is a tuple: (encode_memory/encoding, hidden state, memory cell).
        This method will directly modify this parameter after decoding.
        
        :param inputs: a list of word_id for this step as input.

        :return: the logits after decoding.
        '''
        batch_size = len(decode_memory_list)
        encode_memory, h, c = list(zip(*decode_memory_list))
        enc = torch.stack(encode_memory)
        h   = torch.stack(h)
        c   = torch.stack(c)
        inputs = torch.tensor(inputs, dtype=torch.int, device=self.device)
        # decode for one step using decode_step
        h, c, outputs = self.decoder.decode_step(enc, h, c, inputs)
        del enc
        for k in range(batch_size):
            decode_memory_list[k] = (encode_memory[k], h[k], c[k])
        return outputs

    def beam_decode(self, encode_memory: Tensor):
        return super().beam_decode(encode_memory)