import NetModel_Transformer as tfm
import torch
from data_gen import Img2SeqDataset, get_dataLoader
from utils import root
import time, os

BATCH_SIZE = 8
EPOCHS = 10
PAD_ID = 0
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)


def test_dataLoader():
    data = Img2SeqDataset(root, annotations_file="train_set_labels.csv", img_dir="processed_data")
    dataLoader = get_dataLoader(data, batch_size=4, mode='Img2Seq')
    idx, (img, seq, seq_l) = next(enumerate(dataLoader))
    print(img.shape)
    print(seq[0][7], seq.is_contiguous())
    print(seq_l)
    print(idx)

def test_transformer():
    data = Img2SeqDataset(root, annotations_file="train_set_labels.csv", img_dir="processed_data")
    dataLoader = get_dataLoader(data, batch_size=BATCH_SIZE, mode='Transformer')
    model = tfm.Img2SeqTransformer(patch_size=32, max_img_size=512, max_seq_len=250,
                                    num_encoder_layers=6, num_decoder_layers=6,
                                    d_model=512, nhead=4, vocab_size=data.vocab.size)
    model = model.to(device)
    idx, (img, seq) = next(enumerate(dataLoader))
    img = img.to(device)
    seq = seq.to(device)
    scores = model(img, seq)
    print(scores)

def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    size = len(train_iter) * BATCH_SIZE
    start_t = time.time()
    for idx, (img, seq) in enumerate(train_iter):
        img = img.cuda()
        seq = seq.cuda()
        seq_input = seq[:-1, :]
        logits = model(img, seq_input)
        optimizer.zero_grad()
        seq_out = seq[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), seq_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        if idx % 60 == 0:
            end_t = time.time()
            print(f"loss: {loss:>7f}  [{idx * BATCH_SIZE:>5d}/{size:>5d}]; time: {(end_t - start_t):.3f}")
            start_t = time.time()
    return losses / len(train_iter)


def evaluate(model, val_iter):
    model.eval()
    losses = 0
    size = len(val_iter) * BATCH_SIZE
    start_t = time.time()
    for idx, (img, seq) in (enumerate(val_iter)):
        img = img.cuda()
        seq = seq.cuda()

        seq_input = seq[:-1, :]

        logits = model(img, seq_input)
        seq_out = seq[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), seq_out.reshape(-1))
        losses += loss.item()
        if idx % 60 == 0:
            end_t = time.time()
            print(f"loss: {loss:>7f}  [{idx * BATCH_SIZE:>5d}/{size:>5d}]; time: {(end_t - start_t):.3f}")
            start_t = time.time()
    return losses / len(val_iter)

if __name__ == '__main__':
    train_set = Img2SeqDataset(root, annotations_file="train_set_labels.csv", img_dir="prcd_data/train")
    train_iter = get_dataLoader(train_set, batch_size=BATCH_SIZE, mode='Transformer')
    val_set = Img2SeqDataset(root, annotations_file="val_set_labels.csv", img_dir="prcd_data/validate")
    val_iter = get_dataLoader(val_set, batch_size=BATCH_SIZE, mode='Transformer')
    transformer = tfm.Img2SeqTransformer(patch_size=32, max_img_size=512, max_seq_len=200,
                                    num_encoder_layers=8, num_decoder_layers=8,
                                    d_model=512, nhead=8, vocab_size=train_set.vocab.size)
    if os.path.isfile("transformer_weights.pth"):
        print("Load the weights")
        transformer.load_state_dict(torch.load("transformer_weights.pth"))
    transformer = transformer.to(device)
    optimizer = torch.optim.Adam(
        transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
    )
    for epoch in range(0, EPOCHS):
        start_time = time.time()
        train_loss = train_epoch(transformer, train_iter, optimizer)
        end_time = time.time()
        print((f"Epoch: {epoch + 1}, Train loss: {train_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"))
    with torch.no_grad():
        val_loss = evaluate(transformer, val_iter)
    print(f"Val loss: {val_loss:.3f}")
    torch.save(transformer.state_dict(), "transformer_weights.pth")