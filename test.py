import model.Transformer as tfm
import torch
import torch.nn as nn
from data_gen import Img2SeqDataset, get_dataLoader
import time, os
from torchvision import models
from pkg.utils.vocab import vocab

BATCH_SIZE = 90
EPOCHS = 5
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2

root = os.getcwd()
data_dir = 'data/prcd_data_small'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device_ids = range(torch.cuda.device_count())
multi_gpu = True if len(device_ids) > 1 else False

_vocab = vocab(root)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
transformer = tfm.Img2SeqTransformer(feature_size=None, extractor_name='resnet34', max_seq_len=200,
                                    tr_extractor=True, num_encoder_layers=6, num_decoder_layers=6,
                                    d_model=512, nhead=8, vocab_size=_vocab.size,
                                    dim_feedforward=1024, dropout=0.2)
if os.path.isfile("model weights/transformer_weights.pth"):
    print("Load the weights")
    transformer.load_state_dict(torch.load("model weights/transformer_weights.pth"))


def test_dataLoader():
    data = Img2SeqDataset(root=root, data_dir=data_dir, img_dir="train", annotations_file="train_set_labels.csv")
    dataLoader = get_dataLoader(data, batch_size=4, mode='Img2Seq')
    idx, (img, seq, seq_l) = next(enumerate(dataLoader))
    print(img.shape)
    print(seq[0][7], seq.is_contiguous())
    print(seq_l)
    print(idx)

def test_transformer():
    data = Img2SeqDataset(root=root, data_dir=data_dir, img_dir="train", annotations_file="train_set_labels.csv")
    dataLoader = get_dataLoader(data, batch_size=BATCH_SIZE, mode='Transformer')
    model = tfm.Img2SeqTransformer(feature_size=(8, 16), extractor_name='resnet34', max_seq_len=200,
                                    tr_extractor=False, num_encoder_layers=6, num_decoder_layers=6,
                                    d_model=512, nhead=8, vocab_size=data.vocab.size)
    model = model.to(device)
    (img, seq) = next(iter(dataLoader))
    img = img.to(device)
    seq = seq.to(device)
    scores = model(img, seq)
    print(seq[:, 0])
    print(scores)

def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    size = len(train_iter) * BATCH_SIZE
    start_t = time.time()
    threshold = 4000 // BATCH_SIZE
    for idx, (img, seq) in enumerate(train_iter):
        img = img.to(device)
        seq = seq.to(device)
        seq_input = seq[:, :-1]
        logits = model(img, seq_input)  # (batch_size, lenth, vocab_size)
        optimizer.zero_grad()
        seq_out = seq[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), seq_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        if idx % threshold == 0:
            end_t = time.time()
            print(f"loss: {loss.item():>7f}  [{idx * BATCH_SIZE:>5d}/{size:>5d}]; time: {(end_t - start_t):.3f}")
            print("  Our output:", _vocab.decode(greedy_decode(logits[0, :, :])))
            print("Ground Truth:", _vocab.decode(seq[0, :]))
            start_t = time.time()
    return losses / len(train_iter)

def greedy_decode(scores):
    _, seq = torch.max(scores, dim=1)
    return seq

def evaluate(model, val_iter):
    model.eval()
    losses = 0
    for idx, (img, seq) in (enumerate(val_iter)):
        img = img.to(device)
        seq = seq.to(device)
        seq_input = seq[:-1, :]
        logits = model(img, seq_input)
        seq_out = seq[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), seq_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_iter)

def test_train_transformer():
    train_set = Img2SeqDataset(root=root, data_dir=data_dir, img_dir="train", annotations_file="train_set_labels.csv")
    train_iter = get_dataLoader(train_set, batch_size=BATCH_SIZE, mode='Transformer')
    val_set = Img2SeqDataset(root=root, data_dir=data_dir, img_dir="validate", annotations_file="val_set_labels.csv")
    val_iter = get_dataLoader(val_set, batch_size=BATCH_SIZE, mode='Transformer')
    global transformer
    if multi_gpu:
        transformer = nn.DataParallel(transformer, device_ids = device_ids)
    transformer = transformer.to(device)
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )
    last_val_loss = 10
    for epoch in range(0, EPOCHS):
        start_time = time.time()
        train_loss = train_epoch(transformer, train_iter, optimizer)
        end_time = time.time()
        print((f"Epoch: {epoch + 1}, Train loss: {train_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"))
        with torch.no_grad():
            val_loss = evaluate(transformer, val_iter)
        print(f"Val loss: {val_loss:.3f}")
        if val_loss > last_val_loss:
            break
        last_val_loss = val_loss
        torch.save(transformer.state_dict(), "model weights/transformer_weights.pth")

def test_FeaturesExtractor():
    train_set = Img2SeqDataset(root=root, data_dir=data_dir, img_dir="train", annotations_file="train_set_labels.csv")
    train_iter = get_dataLoader(train_set, batch_size=BATCH_SIZE, mode='Transformer')
    img, seq = next(iter(train_iter))
    extractor = tfm.FeaturesExtractor()
    with torch.no_grad():
        y = extractor(img)
    print(seq[:, 0])

def test_model():
    val_set = Img2SeqDataset(root=root, data_dir=data_dir, img_dir="validate", annotations_file="val_set_labels.csv")
    '''
    if os.path.isfile("model weights/transformer_weights.pth"):
        print("Load the weights")
        transformer.load_state_dict(torch.load("model weights/transformer_weights.pth"))
    else:
        raise Exception('Cannot load the model')
    '''
    import Levenshtein as lvt
    from tqdm import tqdm
    global transformer
    transformer = transformer.to(device)
    transformer.eval()
    t_lvt_d = 0
    for i in tqdm(range(len(val_set))):
        img, seq = val_set[i]
        img = img.unsqueeze(0)
        output = predict(img, transformer)
        grd_truth = _vocab.decode(seq)
        lvt_d = lvt.distance(output, grd_truth)
        t_lvt_d += lvt_d
        if i % 500 == 0:
            print("  Our Output:", output)
            print("Ground Truth:", grd_truth)
            print("Levenshtein distance:", lvt_d)
    print("Levenshtein mean:", float(t_lvt_d) / len(val_set))

def predict(img, model, max_len=200):
    img = img.to(device)
    memory = model.encode(img)
    seq = torch.ones(1, 1).fill_(SOS_ID).type(torch.long).to(device)
    for i in range(max_len - 1):
        out = model.decode(seq, memory)
        out = out.transpose(0, 1)
        scores = model.generator(out[:, -1])
        _, next_word = torch.max(scores, dim = 1)
        next_word = next_word.item()
        seq = torch.cat([seq, torch.ones(1, 1).type_as(img.data).fill_(next_word)], dim=0)
        if next_word == EOS_ID:
            break
    return _vocab.decode(seq)

def num_param():
    transformer = tfm.Img2SeqTransformer(feature_size=None, extractor_name='resnet34', max_seq_len=200,
                                    tr_extractor=False, num_encoder_layers=6, num_decoder_layers=6,
                                    d_model=512, nhead=8, vocab_size=_vocab.size,
                                    dim_feedforward=1024, dropout=0.2)
    num = 0
    for p in transformer.parameters():
        n = 1
        for d in p.shape:
            n *= d
        num += n
    print(num)


if __name__ == '__main__':
    test_train_transformer()