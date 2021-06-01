from typing import Any, Union
import torch
from torch.nn.functional import softmax
from torch.tensor import Tensor

from pkg.utils.BeamSearch import BeamSearch, PAD_ID
from pkg.utils.utils import split_list, flatten_list
from model.Transformer import TransformerType


class BeamSearchTransformer(BeamSearch):
    """
    BeamSearch class for transformer model.
    """

    def __init__(self, tf_model: TransformerType, device: str='cpu', beam_width: int=10, topk: int=1, 
                max_len: int=200, max_batch: int=1):
        super(BeamSearchTransformer, self).__init__(device=device, beam_width=beam_width, topk=topk, 
                                                    max_len=max_len, max_batch=max_batch)
        self.model = tf_model

    def init_decode_memory(self, encode_memory: Tensor):
        '''
        Initialize the decode_memory by encode_memory using decoder.init_hidden_states.
        The decode memory is a tuple: (encode_memory/encoding, hidden state, memory cell)
        
        :param encode_memory: encode_memory, or encodings
        :return: a list of decode memory for each batch
        '''
        decode_memory = self.model.init_decode_memory(encode_memory)
        padding_mask = None
        return [decode_memory, padding_mask]
        

    def decode_step(self, decode_memory: 'list[list[list[Tensor]]]', inputs: Tensor) -> Tensor:
        '''
        Decode for single step using Img2SeqTransformer.decode_step.

        :param decode_memory_list: a list of decode memory for decode_step.
        The decode memory is list[list[Tensor]], the outer list stores decode memory for each decode layer,
        and the inner list is the decode memory: [self_attn_key, self_attn_value, src_attn_key, src_attn_value].
        This method will directly modify this parameter after decoding.
        
        :param inputs: a list of word_id for this step as input.

        :return: the logits after decoding.
        '''
        memory = decode_memory[0]
        padding_mask = decode_memory[1]
        cur_padding_mask = (inputs == PAD_ID).unsqueeze(-1)
        if padding_mask is None:
            padding_mask = cur_padding_mask
        else:
            padding_mask = torch.cat((padding_mask, cur_padding_mask), dim=-1)
        # decode for one step using decode_step
        outputs = self.model.decode_step(seq=inputs, decode_memory=memory, pos=None, tgt_padding_mask=padding_mask)
        decode_memory[0] = memory
        decode_memory[1] = padding_mask
        return outputs

    def trans_decode_memory(self, decode_memory: 'list[list[list[Tensor]]]', indices: Tensor) -> None:
        memory = decode_memory[0]
        memory = flatten_list(memory)
        memory = [m[indices] for m in memory]
        memory = split_list(memory, d=4)
        decode_memory[0] = memory
        decode_memory[1] = decode_memory[1][indices]

    def beam_decode(self, encode_memory: Tensor) -> Tensor:
        ans = super().beam_decode(encode_memory)
        self.model.clear_cache()
        return ans

    def greedy_decode(self, encode_memory: Tensor) -> Tensor:
        ans = super().greedy_decode(encode_memory)
        self.model.clear_cache()
        return ans

    def sample(self, encode_memory: Tensor, gts: Tensor, forcing_num: int, vocab_size: int) -> 'tuple[Tensor, list[Tensor]]':
        return super().sample(encode_memory=encode_memory, gts=gts, forcing_num=forcing_num, vocab_size=vocab_size)

    """
    
    def split_decode_memory(self, decode_memory: 'list[list[Tensor]]', batch_size: int) -> 'list[list[list[Tensor]]]':
        decode_memory_list = flatten_list(decode_memory)
        decode_memory_list = [[memory[k] for k in range(batch_size)] for memory in decode_memory_list]
        decode_memory_list = list(zip(*decode_memory_list))
        decode_memory_list = [split_list(l=list(x), d=4) for x in decode_memory_list]
        return decode_memory_list

    def merge_decode_memory(self, decode_memory_list: 'list[list[list[Tensor]]]'):
        decode_memory = [flatten_list(memory) for memory in decode_memory_list]
        decode_memory = list(zip(*decode_memory))
        decode_memory = [torch.stack(x) for x in decode_memory]
        decode_memory = split_list(l=decode_memory, d=4)
        return decode_memory

    """