import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import utils
from utils import PAD_ID

BATCH_SIZE = 64

def generate_batch_transformer(data_batch):
    img_batch, seq_batch = list(zip(*data_batch))
    seq_batch = pad_sequence(seq_batch, padding_value=PAD_ID)
    return img_batch, seq_batch

def generate_batch_Img2Seq(data_batch):
    """
    generate the batch in decending order by sequence lenths.
    """
    data_batch.sort(key=(lambda data: len(data[1])), reverse=True)
    img_batch, seq_batch = list(zip(*data_batch))
    seq_batch = pad_sequence(seq_batch, padding_value=PAD_ID)
    return torch.stack(img_batch), seq_batch

class Img2SeqDataset(Dataset):
    def __init__(self, annotations_file, img_dir):
        self.vocab = utils.vocab()
        self.img_labels = self.vocab.encoding_labels(pd.read_csv(annotations_file))
        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        img_id = self.img_labels.iloc[i, 0]
        label = self.img_labels.iloc[i, 1]
        img = utils.read_img(img_id, self.img_dir)
        return ToTensor(img), ToTensor(label)

def get_dataLoader(dataset, mode='Img2Seq'):
    if mode == 'Transformer':
        return DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch_transformer)
    elif mode == 'Img2Seq':
        return DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch_Img2Seq)
    else:
        raise Exception('Unknown mode')