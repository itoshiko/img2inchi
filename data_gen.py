import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import utils

PAD_ID = None
vocab_size = None

BATCH_SIZE = 2

def generate_batch_transformer(data_batch):
    img_batch, seq_batch = list(zip(*data_batch))
    seq_batch = pad_sequence(seq_batch, padding_value=PAD_ID)
    encoded_seq_batch = utils.one_hot(seq_batch, vocab_size)
    return torch.stack(img_batch), encoded_seq_batch

def generate_batch_Img2Seq(data_batch):
    """
    generate the batch in decending order by sequence lenths.
    """
    data_batch.sort(key=(lambda data: len(data[1])), reverse=True)
    img_batch, seq_batch = list(zip(*data_batch))
    seq_lenth = [len(seq) for seq in seq_batch]
    seq_batch = pad_sequence(seq_batch, padding_value=PAD_ID)
    seq_batch = seq_batch.transpose(0, 1)
    encoded_seq_batch = utils.one_hot(seq_batch, vocab_size)
    return torch.stack(img_batch), encoded_seq_batch, seq_lenth

class Img2SeqDataset(Dataset):
    def __init__(self, root, annotations_file, img_dir):
        self.vocab = utils.vocab()
        global PAD_ID, vocab_size
        PAD_ID = self.vocab.PAD_ID
        vocab_size = self.vocab.size
        annotations_file = utils.join_path(root, annotations_file)
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = utils.join_path(root, img_dir)
        self.img_trans = ToTensor()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        img_id = self.img_labels.iloc[i, 0]
        label = self.img_labels.iloc[i, 1]
        img = utils.read_img(img_id, self.img_dir)
        img = self.img_trans(img).squeeze(0)
        label = self.vocab.tokenizer(label)
        return img, label

def get_dataLoader(dataset, mode='Img2Seq'):
    if mode == 'Transformer':
        return DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch_transformer)
    elif mode == 'Img2Seq':
        return DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch_Img2Seq)
    else:
        raise Exception('Unknown mode')