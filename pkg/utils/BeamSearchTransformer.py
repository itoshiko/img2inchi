
from torch import stack, topk
from torch.cuda import memory
from torch.nn.functional import softmax
from torch.nn import Module
from torch.tensor import Tensor

from pkg.utils.BeamSearch import BeamSearch
from pkg.utils.utils import split_list, flatten_list
from model.Transformer import TransformerType


class BeamSearchTransformer(BeamSearch):
    """
    BeamSearch class for transformer model.
    """

    def __init__(self, tf_model: TransformerType, device, beam_width: int=10, topk: int=1, max_len: int=200, max_batch: int=1):
        super(BeamSearchTransformer, self).__init__(beam_width, topk, max_len, max_batch)
        self.model = tf_model
        self.device = device

    def split_decode_memory(self, decode_memory: 'list[list[Tensor]]', batch_size: int) -> 'list[list[list[Tensor]]]':
        decode_memory_list = flatten_list(decode_memory)
        decode_memory_list = [[memory[k] for k in range(batch_size)] for memory in decode_memory_list]
        decode_memory_list = list(zip(*decode_memory_list))
        decode_memory_list = [split_list(l=x, d=4) for x in decode_memory_list]
        return decode_memory_list

    def merge_decode_memory(self, decode_memory_list: 'list[list[list[Tensor]]]'):
        decode_memory = [flatten_list(memory) for memory in decode_memory_list]
        decode_memory = list(zip(*decode_memory))
        decode_memory = [stack(x) for x in decode_memory]
        decode_memory = split_list(l=decode_memory, d=4)
        return decode_memory

    def init_decode_memory(self, encode_memory: Tensor):
        '''
        Initialize the decode_memory by encode_memory using decoder.init_hidden_states.
        The decode memory is a tuple: (encode_memory/encoding, hidden state, memory cell)
        
        :param encode_memory: encode_memory, or encodings
        :return: a list of decode memory for each batch
        '''
        batch_size = encode_memory.shape[0]
        decode_memory = self.model.init_decode_memory(encode_memory)
        return self.split_decode_memory(decode_memory, batch_size)
        

    def decode_step(self, decode_memory_list: 'list[list[list[Tensor]]]', inputs: 'list[int]') \
        -> 'list[list[tuple[list[list[Tensor]], int, float]]]':
        '''
        Decode for single step using Img2SeqTransformer.decode_step.

        :param decode_memory_list: a list of decode memory for decode_step.
        The decode memory is list[list[Tensor]], the outer list stores decode memory for each decode layer,
        and the inner list is the decode memory: [self_attn_key, self_attn_value, src_attn_key, src_attn_value].
        
        :param inputs: a list of word_id for this step.

        :return: list[list[tuple]]. Outer list corresponding to the decode answer for each input.
        Inner list gives the top beam_width answer. Each answer is a tuple:
        (new_decode_memory, new_word_id, conditional_probability)
        '''
        batch_size = len(decode_memory_list)
        decode_memory = self.merge_decode_memory(decode_memory_list)
        inputs = Tensor(inputs).long().to(self.device)
        # decode for one step using decode_step
        outputs = self.model.decode_step(seq=inputs, decode_memory=decode_memory, pos=None, tgt_padding_mask=None)
        decode_memory_list = self.split_decode_memory(decode_memory)
        probs = softmax(outputs, dim=1)
        probs, indexes = topk(probs, self.beam_width, dim=1)
        decode_answers = [
            [(decode_memory_list[k], int(indexes[j].item), float(probs[k][j].item)) 
            for j in range(self.beam_width)] 
            for k in range(batch_size)
        ]
        return decode_answers

    def beam_decode(self, encode_memory: Tensor):
        ans = super().beam_decode(encode_memory)
        self.model.clear_cache()
        return ans