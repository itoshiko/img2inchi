import pandas as pd
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from pkg.utils.utils import read_img, join
from pkg.utils.vocab import vocab


class Img2SeqDataset(Dataset):
    def __init__(self, root, data_dir, img_dir, annotations_file, vocab):
        self.vocab = vocab
        annotations_file = join(root, data_dir, annotations_file)
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels['InChI'] = self.vocab.encode_all(self.img_labels)
        self.img_dir = join(root, data_dir, img_dir)
        self.img_trans = ToTensor()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, i):
        img_id = self.img_labels.iloc[i, 0]
        seq = self.img_labels.iloc[i, 1]
        # img = self.img_trans(img).squeeze(0)
        img = self.img_trans(read_img(root=self.img_dir, img_id=img_id, mode='GRAY'))
        seq = Tensor(seq).long()
        return img, seq

    def generate_batch_transformer(self, data_batch):
        img_batch, seq_batch = list(zip(*data_batch))
        img_batch = torch.stack(img_batch)
        img_batch = img_batch * torch.ones([1, 3, 1, 1])
        seq_batch = pad_sequence(seq_batch, padding_value=self.vocab.PAD_ID).transpose(0, 1).contiguous()
        return img_batch, seq_batch.long()

    def generate_batch_lstm(self, data_batch):
        """
        generate the batch in decending order by sequence lenths.
        """
        data_batch.sort(key=(lambda data: len(data[1])), reverse=True)
        img_batch, seq_batch = list(zip(*data_batch))
        seq_batch = pad_sequence(seq_batch, padding_value=self.vocab.PAD_ID).transpose(0, 1).contiguous()
        return torch.stack(img_batch), seq_batch.long()


def get_dataLoader(dataset: Img2SeqDataset, batch_size: int = 4, model_name='transformer'):
    if model_name == 'transformer':
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=dataset.generate_batch_transformer)
    elif model_name == 'lstm':
        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, collate_fn=dataset.generate_batch_lstm)
    else:
        raise Exception('Unknown mode')
