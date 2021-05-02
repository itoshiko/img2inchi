import os
import numpy as np
import torch
from torch import Tensor
import cv2

root = "D:/Tsinghua/2021.2/Artificial_Intelligence/Final Project/data"
PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'

class vocab():
    def __init__(self):
        self.vocab_to_int, self.int_to_vocab = self.get_vocab()
        self.size = len(self.vocab_to_int)
        self.PAD_ID = self.__call__(PAD)
        self.SOS_ID = self.__call__(SOS)
        self.EOS_ID = self.__call__(EOS)

    def tokenizer(self, inchi):
        if not (inchi[:8] == "InChI=1S"):
            raise Exception("Not Matching 'InChI=1S'")
        inchi = inchi[8:]
        res = []
        res.append(self.vocab_to_int[SOS])
        for c in inchi:
            res.append(self.vocab_to_int[c])
        return Tensor(res).long()

    def translate(self, seq):
        if len(seq.shape) == 2:
            if seq.shape[1] > 1:
                raise Exception("Too many input sequences")
            else:
                seq = seq.squeeze(1)
        elif len(seq.shape) > 2:
            raise Exception("Dimension is wrong")
        inchi = "InChI=1S"
        for token in seq:
            if token == self.SOS_ID:
                continue
            elif token == self.EOS_ID:
                break
            elif token == self.PAD_ID:
                break
            else:
                inchi += self.__call__(int(token))
        return inchi

    def get_vocab(self):
        import pickle
        with open(join_path(root, "vocab_to_int.pkl"), "rb") as f:
            vocab_to_int = pickle.load(f)
        with open(join_path(root, "int_to_vocab.pkl"), "rb") as f:
            int_to_vocab = pickle.load(f)
        return vocab_to_int, int_to_vocab
    
    def __call__(self, x):
        if isinstance(x, int):
            return self.int_to_vocab[x]
        if isinstance(x, str):
            return self.vocab_to_int[x]
        return None

def join_path(path, *subdirs):
    for dir in subdirs:
        path = os.path.join(path, dir)
    return path

def get_img_path(img_id, path):
    return join_path(path, img_id[0], img_id[1], img_id[2], f'{img_id}.png')

def read_img(img_id, path):
    '''
    read image by cv2
    '''
    img_path = get_img_path(img_id, path)
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path)
    return img

def one_hot(seqs: Tensor, vocab_size: int):
    d1, d2 = seqs.shape
    seqs = seqs.unsqueeze(-1)
    one_hot = torch.zeros(d1, d2, vocab_size)
    one_hot.scatter_(dim=2, index=seqs.long(), src=torch.ones(d1, d2, vocab_size))
    return one_hot

if __name__ == '__main__':
    vocab = vocab()
    seq = Tensor([9, 7, 5, 34, 12, 1])
    print(vocab.translate(seq))