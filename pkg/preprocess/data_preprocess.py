import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import cv2
import os
import threading
from ..utils.utils import join, create_dirs
from ..utils.vocab import build_vocabulary
from .img_process import prc_img


def read_data_set(root, dir_name, file_name):
    train_set = pd.read_csv(join(root, dir_name, file_name), 
                        dtype={ 'image_id': 'string', 'InChI': 'string' })
    return train_set

def get_max_len(data_set):
    data_set['InChI_len'] = data_set['InChI'].apply(len).astype(np.uint16)
    max_len = data_set['InChI_len'].max()
    return max_len

def train_val_split(root, dir_name, data_set, train_size, val_size):
    val_set = data_set.iloc[-val_size:].reset_index(drop=True)
    val_set.to_csv(join(root, dir_name, 'val_set_labels.csv'), index=0)
    s = train_size if train_size > 0 else -val_size 
    train_set = data_set.iloc[:s].reset_index(drop=True)
    train_set.to_csv(join(root, dir_name, 'train_set_labels.csv'), index=0)
    return val_set, train_set

def create_data_dirs(root, dir_name, name):
    l = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    data_rt = join(root, dir_name, name)
    create_dirs(data_rt)
    for d1 in l:
        create_dirs(data_rt, d1)
        for d2 in l:
            create_dirs(data_rt, d1, d2)
            for d3 in l:
                create_dirs(data_rt, d1, d2, d3)
    return data_rt

def prc_imgs(root, origin_dir_name, prcd_dir_name, name, img_list, num_threads=8):
    origin_root = join(root, origin_dir_name, 'train')
    prcd_root = create_data_dirs(root, prcd_dir_name, name)
    sz = len(img_list) // num_threads + 1
    prc_threads = [prc_img_thread(i // sz, origin_root, prcd_root, img_list[i:i + sz]) for i in range(0, len(img_list), sz)]
    for thread in prc_threads:
        thread.start()
    for thread in prc_threads:
        thread.join()


class prc_img_thread(threading.Thread):
    def __init__(self, threadID, origin_root, prcd_root, img_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.img_list = img_list
        self.origin_root = origin_root
        self.prcd_root = prcd_root
    
    def run(self):
        origin_root = self.origin_root
        prcd_root = self.prcd_root
        print(f"Thread-{self.threadID}: start processing images.")
        for i, img_id in enumerate(self.img_list):
            prc_img(img_id, source_root=origin_root, target_root=prcd_root)
            if i % 10000 == 0:
                print(f"Thread-{self.threadID}: processing images: [{i}/{len(self.img_list)}]")
        print(f"Thread-{self.threadID}: finish processing images.")
        
