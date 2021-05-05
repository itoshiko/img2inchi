import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import cv2
import os
from ..utils.utils import join, create_dirs
from ..utils.vocab import build_vocabulary
from .img_process import prc_img

data_root = "D:/Tsinghua/2021.2/Artificial_Intelligence/Final Project/data"

VAL_SIZE = int(100e3)
TRAIN_SIZE = -1

prcd_data_root = 'prcd_data'

def read_data_set(name):
    train_set = pd.read_csv(join(data_root, name), 
                        dtype={ 'image_id': 'string', 'InChI': 'string' })
    return train_set

def get_max_len(data_set):
    data_set['InChI_len'] = data_set['InChI'].apply(len).astype(np.uint16)
    max_len = data_set['InChI_len'].max()
    return max_len

def train_val_split(data_set):
    val_set = data_set.iloc[-VAL_SIZE:].reset_index(drop=True)
    val_set.to_csv(join(data_root, 'val_set_labels.csv'), index=0)
    s = TRAIN_SIZE if TRAIN_SIZE > 0 else -VAL_SIZE 
    train_set = data_set.iloc[:s].reset_index(drop=True)
    train_set.to_csv(join(data_root, 'train_set_labels.csv'), index=0)
    return val_set, train_set

def create_data_dirs(name):
    l = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    data_rt = join(data_root, prcd_data_root, name)
    create_dirs(data_rt)
    for d1 in l:
        create_dirs(data_rt, d1)
        for d2 in l:
            create_dirs(data_rt, d1, d2)
            for d3 in l:
                create_dirs(data_rt, d1, d2, d3)
    return data_rt

def prc_imgs(data, name):
    data_root = create_data_dirs(name)
    l = len(data) 
    for i in tqdm(range(l)):
        prc_img(data.loc[i, 'image_id'], target_folder=data_root)

