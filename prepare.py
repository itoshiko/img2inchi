import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import cv2
import os
from utils import root, join_path


if_gen_vocab = True
if_preprocessed = True
if_split = True

DEBUG = False
MAX_INCHI_LEN = 200
IMG_WIDTH = 1024
IMG_HEIGHT = 512
THRESHOLD = 50
VAL_SIZE = int(40e3)
TRAIN_SIZE = int(120e3)
CHUNK_SIZE = int(40e3)
SOS = '<SOS>'
EOS = '<EOS>'
PAD = '<PAD>'
vocab_to_int = None
int_to_vocab = None


def read_train_set():
    train_set = pd.read_csv(join_path(root, 'train_labels.csv'), 
                        dtype={ 'image_id': 'string', 'InChI': 'string' })
    return train_set

def build_vocabulary(train_set):
    global vocab_to_int, int_to_vocab, if_gen_vocab
    if not if_gen_vocab:
        try:
            with open(join_path(root, "vocab_to_int.pkl"), "rb") as f:
                global vocab_to_int
                vocab_to_int = pickle.load(f)
            with open(join_path(root, "int_to_vocab.pkl"), "rb") as f:
                global int_to_vocab
                int_to_vocab = pickle.load(f)
            return
        except:
            if_gen_vocab = True
    tokens = [PAD, SOS, EOS]
    vocabulary = set()
    for s in tqdm(train_set['InChI'].values):
        vocabulary.update(s)
    vocabulary = list(vocabulary)
    vocabulary.sort()
    vocabulary = tokens + vocabulary
    vocab_to_int = dict(zip(vocabulary, np.arange(len(vocabulary), dtype=np.uint8)))
    int_to_vocab = dict(zip(np.arange(len(vocabulary), dtype=np.uint8), vocabulary))
    with open(join_path(root, "vocab_to_int.pkl"), "wb") as f:
        pickle.dump(vocab_to_int, f)
    with open(join_path(root, "int_to_vocab.pkl"), "wb") as f:
        pickle.dump(int_to_vocab, f)

def to_int(inchi):
    if not (inchi[:8] == "InChI=1S"):
        raise Exception("Not Matching 'InChI=1S'")
    inchi = inchi[8:]
    res = []
    res.append(vocab_to_int[SOS])
    for c in inchi:
        res.append(vocab_to_int[c])
    return np.array(res, dtype=np.uint8)

def encoding_lables(data):
    data['InChI'] = data['InChI'].apply(to_int)
    return data

def preprocess_train_set(train_set):
    if if_preprocessed:
        try:
            train_set = pd.read_csv('preprocessed_train_labels.csv',
                                    dtype={ 'image_id': 'string', 'InChI': 'string' })
            return train_set
        except:
            pass
    train_set['InChI_len'] = train_set['InChI'].apply(len).astype(np.uint16)
    train_set = train_set.loc[train_set['InChI_len'] <= MAX_INCHI_LEN + 8 - 2].reset_index(drop = True)
    del train_set['InChI_len']
    train_set.to_csv('preprocessed_train_labels.csv', index=0)
    print("Finished saving")
    return train_set

def train_val_split(train_set):
    if if_split:
        try:
            val_set = pd.read_csv('val_set_labels.csv',
                                    dtype={ 'image_id': 'string', 'InChI': 'string' })
            train_set = pd.read_csv('train_set_labels.csv',
                                    dtype={ 'image_id': 'string', 'InChI': 'string' })
            return val_set, train_set
        except:
            pass
    val_set = train_set.iloc[-VAL_SIZE:].reset_index(drop=True)
    val_set.to_csv('val_set_labels.csv', index=0)
    s = TRAIN_SIZE if TRAIN_SIZE > 0 else -VAL_SIZE 
    train_set = train_set.iloc[:s].reset_index(drop=True)
    train_set.to_csv('train_set_labels.csv', index=0)
    return val_set, train_set

def create_dirs(path, *subdirs):
    if len(subdirs) > 0:
        path = join_path(path, *subdirs)
    if not os.path.exists(path):
        os.makedirs(path)

def create_data_dirs():
    l = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    data_root = join_path(root, "processed_data")
    create_dirs(data_root)
    for d1 in l:
        create_dirs(data_root, d1)
        for d2 in l:
            create_dirs(data_root, d1, d2)
            for d3 in l:
                create_dirs(data_root, d1, d2, d3)

def pad_resize(img):
    h, w = img.shape
    '''
    if w <= IMG_WIDTH and h <= IMG_HEIGHT:
        pad_v = IMG_WIDTH - w
        pad_h = IMG_HEIGHT - h
    else:
        pad_h, pad_v = 0, 0
        hw_ratio = (h / w) - (IMG_HEIGHT / IMG_WIDTH)
        if hw_ratio < 0:
            pad_h = int(abs(hw_ratio) * w)
        else:
            wh_ratio = (w / h) - (IMG_WIDTH / IMG_HEIGHT)
            pad_v = int(abs(wh_ratio) * h)

    img = np.pad(img, [(pad_h // 2, pad_h - pad_h // 2), (pad_v // 2, pad_v - pad_v // 2)], mode='constant')
    '''
    h, w = img.shape
    pad_h, pad_v = 0, 0
    hw_ratio = (h / w) - (IMG_HEIGHT / IMG_WIDTH)
    if hw_ratio < 0:
        pad_h = int(abs(hw_ratio) * w / 2)
    else:
        wh_ratio = (w / h) - (IMG_WIDTH / IMG_HEIGHT)
        pad_v = int(abs(wh_ratio) * h // 2)

    img = np.pad(img, [(pad_h, pad_h), (pad_v, pad_v)], mode='constant')
    img = cv2.resize(img,(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)

    return img

def process_img(img_id, source_folder="train", target_folder="processed_data"):
    source_file_path =  join_path(root, source_folder, img_id[0], img_id[1], img_id[2], f'{img_id}.png')
    target_file_path =  join_path(root, target_folder, img_id[0], img_id[1], img_id[2], f'{img_id}.png')
    img = 255 - cv2.imread(source_file_path, cv2.IMREAD_GRAYSCALE)
    
    # rotate counter clockwise to get horizontal images
    h, w = img.shape
    if h > w:
        img = np.rot90(img)
    img = pad_resize(img)
    img = (img / img.max() * 255).astype(np.uint8)
    img[np.where(img > THRESHOLD)] = 255
    img[np.where(img <= THRESHOLD)] = 0
    cv2.imwrite(target_file_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    if DEBUG:
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

if __name__ == "__main__":
    train_set = read_train_set()
    build_vocabulary(train_set)
    print(vocab_to_int)
    '''
    if not if_split:
        train_set = preprocess_train_set(train_set)
        print(train_set.head(3))
    val_set, train_set = train_val_split(train_set)
    encoding_lables(val_set)
    encoding_lables(train_set)
    print(train_set.info())
    print(val_set.info())
    create_data_dirs()
    l = 10 if DEBUG else len(train_set) 
    for i in tqdm(range(l)):
        process_img(train_set.loc[i, 'image_id'])
    '''
