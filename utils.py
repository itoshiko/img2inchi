import os
import numpy as np
import torch
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

    def encode(self, inchi):
        if not (inchi[:9] == "InChI=1S/"):
            raise Exception("Not Matching 'InChI=1S'")
        inchi = inchi[9:]
        res = []
        res.append(self.vocab_to_int[SOS])
        matched = False
        cur_word = ''
        for c in inchi:
            if matched:
                if cur_word + c in self.vocab_to_int:
                    cur_word += c
                else:
                    res.append(self.vocab_to_int[cur_word])
                    cur_word = c
                    matched = cur_word in self.vocab_to_int
            else:
                cur_word += c
                matched = cur_word in self.vocab_to_int
        if cur_word != '':
            if matched:
                res.append(self.vocab_to_int[cur_word])
            else:
                print(cur_word)
                raise Exception("Cannot match the InChI sequence")
        res.append(self.vocab_to_int[EOS])
        return np.array(res, dtype=np.uint8)

    def encode_all(self, data):
        return data['InChI'].apply(self.encode)

    def decode(self, seq):
        if len(seq.shape) == 2:
            if seq.shape[1] > 1:
                raise Exception("Too many input sequences")
            else:
                seq = seq.squeeze(1)
        elif len(seq.shape) > 2:
            raise Exception("Dimension is wrong")
        inchi = "InChI=1S/"
        for token in seq:
            if token == self.SOS_ID:
                continue
            elif token == self.EOS_ID:
                break
            elif token == self.PAD_ID:
                break
            else:
                inchi += self.int_to_vocab[int(token)]
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

def one_hot(seqs, vocab_size):
    d1, d2 = seqs.shape
    seqs = seqs.unsqueeze(-1)
    one_hot = torch.zeros(d1, d2, vocab_size)
    one_hot.scatter_(dim=2, index=seqs.long(), src=torch.ones(d1, d2, vocab_size))
    return one_hot

if __name__ == '__main__':
    vocab = vocab()
    from torch import Tensor
    seq = Tensor([1, 217, 134, 221, 139, 218, 223, 138, 224, 126, 226, 9, 15, 
                7, 115, 3, 93, 7, 136, 7, 134, 3, 140, 4, 133, 7, 16, 7, 204, 
                7, 127, 3, 126, 4, 142, 7, 133, 4, 141, 7, 132, 7, 193, 7, 138, 
                7, 160, 7, 129, 3, 71, 7, 132, 4, 104, 7, 137, 7, 27, 7, 49, 
                7, 139, 3, 60, 7, 38, 7, 137, 4, 131, 7, 182, 7, 149, 7, 171, 
                7, 130, 3, 135, 4, 82, 7, 131, 10, 138, 7, 16, 6, 71, 7, 82, 6, 
                115, 221, 6, 27, 7, 60, 6, 93, 7, 104, 221, 126, 6, 15, 7, 126, 
                221, 138, 6, 3, 221, 6, 136, 6, 140, 4])
    print(vocab.decode(seq))