import NetModel_Transformer as tfm
from data_gen import Img2SeqDataset, get_dataLoader
from utils import root

def test_dataLoader():
    data = Img2SeqDataset(root, "train_set_labels.csv", "processed_data")
    dataLoader = get_dataLoader(data, 'Img2Seq')
    idx, (img, seq, seq_l) = next(enumerate(dataLoader))
    print(img.shape)
    print(seq[0][7], seq.is_contiguous())
    print(seq_l)
    print(idx)

if __name__ == '__main__':
    test_dataLoader()