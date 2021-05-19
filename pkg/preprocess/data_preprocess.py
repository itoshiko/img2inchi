from time import sleep
from multiprocessing import Process, Pool, RLock, cpu_count
from math import ceil

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.set_lock(RLock())

from pkg.utils.utils import create_dirs, join
from pkg.preprocess.img_process import prc_img


def read_data_set(root, dir_name, file_name):
    train_set = pd.read_csv(join(root, dir_name, file_name),
                            dtype={'image_id': 'string', 'InChI': 'string'})
    return train_set


def get_max_len(data_set):
    data_set['InChI_len'] = data_set['InChI'].apply(len).astype(np.uint16)
    max_len = data_set['InChI_len'].max()
    return max_len


def train_val_split(root, dir_name, data_set, train_size, val_size, config):
    create_dirs(root, dir_name)
    val_set = data_set.iloc[-val_size:].reset_index(drop=True)
    val_set.to_csv(join(root, dir_name, config["val_set_labels"]), index=0)
    s = train_size if train_size > 0 else -val_size
    train_set = data_set.iloc[:s].reset_index(drop=True)
    train_set.to_csv(join(root, dir_name, config["train_set_labels"]), index=0)
    return val_set, train_set


def create_data_dirs(root, dir_name, name):
    l = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    data_rt = join(root, dir_name, name)
    create_dirs(data_rt)
    for d1 in l:
        for d2 in l:
            for d3 in l:
                create_dirs(data_rt, d1, d2, d3)
    return data_rt


def prc_imgs(config, root, origin_dir_name, prcd_dir_name, name, img_list, num_workers=8):
    origin_root = join(root, origin_dir_name, 'train')
    prcd_root = create_data_dirs(root, prcd_dir_name, name)
    num_workers = min(num_workers, cpu_count())
    chunk_size = 50
    pool = Pool(processes=num_workers)
    img_prc = img_processer(chunk_size, origin_root, prcd_root, img_list, config)
    argument_list = range(0, len(img_list), chunk_size)
    tqdm_text = f"Processing Images({name})"
    with tqdm(total=len(img_list), desc=tqdm_text) as progbar:
        for result in pool.imap_unordered(func=img_prc.run, iterable=argument_list):
            progbar.update(result)
    
    '''
    sz = ceil(float(len(img_list)) / float(num_workers))
    processes = [img_multi_process(i // sz, origin_root, prcd_root, img_list[i:i + sz], config) for i in
                   range(0, len(img_list), sz)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    '''


class img_processer(object):
    def __init__(self, chunk_size, origin_root, prcd_root, img_list, config):
        self.chunk_size = chunk_size
        self.img_list = img_list
        self.origin_root = origin_root
        self.prcd_root = prcd_root
        self._config = config
        self.daemon = True

    def run(self, i):
        origin_root = self.origin_root
        prcd_root = self.prcd_root
        # print(f"Worker-{self.process_id}: start processing images.")
        img_list = self.img_list[i:i + self.chunk_size]
        for img_id in img_list:
            prc_img(img_id, source_root=origin_root, target_root=prcd_root, config=self._config)
        return len(img_list)
        # print(f"Worker-{self.process_id}: finish processing images.")