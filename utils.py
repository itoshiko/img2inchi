import os
import numpy as np
import cv2

root = os.getcwd()
SOS = '<SOS>'
EOS = '<EOS>'
PAD = '<PAD>'
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2

class vocab():
    def __init__(self):
        self.vocab_to_int, self.int_to_vocab = self.get_vocab()
        global PAD_ID, SOS_ID, EOS_ID
        PAD_ID = self.__call__(PAD)
        SOS_ID = self.__call__(SOS)
        EOS_ID = self.__call__(EOS)

    def decode(self, inchi):
        if not (inchi[:8] == "InChI=1S"):
            raise Exception("Not Matching 'InChI=1S'")
        inchi = inchi[8:]
        res = []
        res.append(self.vocab_to_int[SOS])
        for c in inchi:
            res.append(self.vocab_to_int[c])
        return np.array(res, dtype=np.uint8)

    def encoding_labels(self, data):
        data['InChI'] = data['InChI'].apply(self.decode)
        return data

    def get_vocab(self):
        import pickle
        with open("vocab_to_int.pkl", "rb") as f:
            vocab_to_int = pickle.load(f)
        with open("int_to_vocab.pkl", "rb") as f:
            int_to_vocab = pickle.load(f)
        return vocab_to_int, int_to_vocab
    
    def __call__(self, x):
        if isinstance(x, int):
            return self.int_to_vocab[x]
        if isinstance(x, str):
            return self.vocab_to_int[x]
        return None

def path_join(path, *subdirs):
    for dir in subdirs:
        path = os.path.join(path, dir)
    return path

def get_img_path(img_id, path):
    return path_join(path, img_id[0], img_id[1], img_id[2], f'{img_id}.png')

def read_img(img_id, path):
    '''
    read image by cv2
    '''
    img_path = get_img_path(img_id, path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img