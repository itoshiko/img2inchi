import pickle
from typing import Union
import numpy as np
from torch import Tensor
from pkg.utils.utils import join, create_dirs


PAD = '<PAD>'
SOS = '<SOS>'
EOS = '<EOS>'

def build_vocabulary(root, vocab_dir, inchi_list):
    import pickle
    from tqdm import tqdm
    tokens = [PAD, SOS, EOS]
    vocabulary = set()
    is_lower_letter = lambda x: 'a' <= x and x <= 'z'
    for inchi in tqdm(inchi_list):
        layers = inchi.split('/')
        del layers[0]
        build_vocab(vocabulary, layers[0], split_others=False)
        del layers[0]
        for string in layers:
            if is_lower_letter(string[0]):
                vocabulary.add('/' + string[0])
                string = string[1:]
            build_vocab(vocabulary, string, split_others=True)
    for i in range(201):
        vocabulary.add(str(i))
    vocabulary = list(vocabulary)
    vocabulary.sort()
    vocabulary = tokens + vocabulary
    vocab_to_int = dict(zip( vocabulary, np.arange(len(vocabulary), dtype=np.uint8) ))
    int_to_vocab = dict(zip( np.arange(len(vocabulary), dtype=np.uint8), vocabulary ))
    create_dirs(root, vocab_dir)
    with open(join(root, vocab_dir, "vocab_to_int.pkl"), "wb") as f:
        pickle.dump(vocab_to_int, f)
    with open(join(root, vocab_dir, "int_to_vocab.pkl"), "wb") as f:
        pickle.dump(int_to_vocab, f)
    return vocab_to_int, int_to_vocab

def build_vocab(vocabulary: set, string: str, split_others: bool):
    is_num = lambda x: '0' <= x and x <= '9'
    is_capital_letter = lambda x: 'A' <= x and x <= 'Z'
    word = ''
    for s in string:
        if is_num(s):
            if word != '':
                if split_others:
                    vocabulary.update(word)
                else:
                    vocabulary.add(word)
                word = ''
        else:
            if not split_others and is_capital_letter(s):
                if word != '':
                    vocabulary.add(word)
                word = ''
            word += s
    if word != '':
        if split_others:
            vocabulary.update(word)
        else:
            vocabulary.add(word)


class vocab():
    def __init__(self, root, vocab_dir):
        '''
        we'll find the vocab files in root/vocab_dir
        :param root: project root. 
        :param vocab_dir: the vocabulary directory.
        '''
        self.root = root
        self.vocab_dir = vocab_dir
        self.vocab_to_int, self.int_to_vocab = self.get_vocab()
        self.size = len(self.vocab_to_int)
        self.vocab_arr = np.array([self.int_to_vocab[key] for key in self.int_to_vocab])
        self.PAD_ID = self.vocab_to_int[PAD]
        self.SOS_ID = self.vocab_to_int[SOS]
        self.EOS_ID = self.vocab_to_int[EOS]
        self.vocab_arr[self.PAD_ID] = ''

    def encode(self, inchi: str, no_mole_fml: bool=False):
        '''
        encode but remove the molecular formula
        '''
        if not (inchi[:9] == "InChI=1S/"):
            raise Exception("Not Matching 'InChI=1S'")
        inchi = inchi[9:]
        if no_mole_fml:
            s = '/'
            layers = inchi.split(s)
            del layers[0]
            inchi = s.join(layers)
            inchi = s + inchi
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

    '''
    def decode(self, seq: Union[np.ndarray, Tensor]):
        
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
    '''

    def decode(self, seqs: Union[np.ndarray, Tensor]) -> np.ndarray:
        if isinstance(seqs, Tensor):
            seqs: np.ndarray = seqs.cpu().numpy()
        if seqs.ndim == 1:
            batch_size = 1
        else:
            batch_size = seqs.shape[0]
            seqs = seqs.reshape(batch_size, -1)
        seqs = seqs.astype(np.int)
        seqs[seqs == self.SOS_ID] = self.PAD_ID
        mask = np.cumsum(seqs == self.EOS_ID, axis=-1) > 0
        seqs[mask] = self.PAD_ID
        seqs = np.reshape(self.vocab_arr[np.reshape(seqs, -1)], (batch_size, -1))
        result = np.apply_along_axis(lambda a: "".join(a), -1, seqs)
        result = np.char.add("InChI=1S/", result)
        return result

    def get_vocab(self):
        root = self.root
        import pickle
        with open(join(root, self.vocab_dir, "vocab_to_int.pkl"), "rb") as f:
            vocab_to_int = pickle.load(f)
        with open(join(root, self.vocab_dir, "int_to_vocab.pkl"), "rb") as f:
            int_to_vocab = pickle.load(f)
        return vocab_to_int, int_to_vocab
    
    def __call__(self, x):
        if isinstance(x, int):
            return self.int_to_vocab[x]
        if isinstance(x, str):
            return self.vocab_to_int[x]
        return None


if __name__ == '__main__':
    root = "D:/Tsinghua/2021.2/Artificial_Intelligence/Final Project/img2inchi"
    vocab = vocab(root)
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