import csv
import os
import threading
from math import ceil
from threading import Thread
from time import time

import click
import cv2
import numpy as np
import torch

from img2inchi import Img2InchiModel
from img2inchi_transformer import Img2InchiTransformerModel
from pkg.utils.general import Config
from pkg.utils.vocab import vocab as vocabulary
from predict import pre_process


def predict_all(model: Img2InchiModel, vocab: vocabulary, path: str, batch_size: int):
    """predict all for commit
    """
    img_list = []
    img_name_list = []
    img_count = 0
    directory, prefix = path
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file[-3:] == "png" or file[-3:] == "jpg":
                _img_path = os.path.join(directory, file)
                img = cv2.imread(_img_path, cv2.IMREAD_GRAYSCALE)
                img = pre_process(img)
                img = np.tile(img, (3, 1, 1))
                img_list.append(img)
                img_name_list.append(file[0:-4])
                img_count += 1
    # split array
    arr_count = img_count // batch_size + 1
    split_array = np.array_split(np.array(img_list), arr_count)
    seq = []
    for arr in split_array:
        img_list = torch.from_numpy(arr).float()
        seq += vocab.decode(model.predict(img_list, mode="beam")).tolist()
    return seq, img_name_list


def walk_all_dir(path):
    name_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    dir_list = []
    for i in name_list:
        for j in name_list:
            for k in name_list:
                sub_dir_path = path + '/' + i + '/' + j + '/' + k
                prefix = i + j + k
                dir_list.append((sub_dir_path, prefix))
    return dir_list


def write_csv(seq_list, name_list, file_path):
    if not (os.path.exists(file_path) and os.path.isfile(file_path)):
        with open(file_path, mode='w', newline='', encoding='utf8') as cf:
            wf = csv.writer(cf)
            wf.writerow(['image_id', 'InChI'])
            wf.writerows(list(map(list, zip(*[name_list, seq_list]))))
    else:
        with open(file_path, mode='a', newline='', encoding='utf8') as cfa:
            wf = csv.writer(cfa)
            wf.writerows(list(map(list, zip(*[name_list, seq_list]))))
    print("CSV written!")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class commit(Thread):
    def __init__(self, idx, model, vocab, dir_list, batch_size, csv_path, lock):
        super(commit, self).__init__()
        self.model = model
        self.idx = idx
        self.vocab = vocab
        self.batch_size = batch_size
        self.csv_path = csv_path
        self.lock = lock
        self.dir_list = dir_list

    def run(self):
        seq_list = []
        name_list = []
        i = 0
        num_folders = len(self.dir_list)
        start_t = time()
        setup_seed(10)
        for directory in self.dir_list:
            small_seq_list, small_name_list = predict_all(self.model, self.vocab, directory, self.batch_size)
            seq_list = seq_list + small_seq_list
            name_list = name_list + small_name_list
            if len(seq_list) > 5000:
                with self.lock:
                    write_csv(seq_list, name_list, self.csv_path)
                seq_list.clear()
                name_list.clear()
            i += 1
            stop_t = time()
            print(f'Thread-{self.idx}: Processed {i}/{num_folders} folders, using time {stop_t - start_t:04.2f}')
            start_t = stop_t
        with self.lock:
            write_csv(seq_list, name_list, self.csv_path)

@click.command()
@click.option('--model_path', default="./model weights/test_save",
              help='Path to trained model')
@click.option('--data_path', default="./images",
              help='Path to data to process')
@click.option('--batch_size', default=64,
              help='number of images to process at the same time')
@click.option('--csv_filename', default="submission.csv",
              help='Filename of generated csv file')
def main(model_path, data_path, batch_size, csv_filename):
    model_file_name = model_path + '/model.ckpt'
    model_config = model_path + '/export_config.yaml'
    csv_path = os.path.join(data_path, csv_filename)
    config = Config(model_config)
    my_vocab = vocabulary(root=config.path_train_root, vocab_dir=config.vocab_dir)
    model_type = config.model_name
    dir_list = walk_all_dir(data_path)
    device_num = torch.cuda.device_count()
    model = [[] for _ in range(device_num)]
    if model_type == "transformer":
        for i in range(torch.cuda.device_count()):
            config.device = 'cuda:' + str(i)
            model[i] = Img2InchiTransformerModel(config, output_dir='', vocab=my_vocab, need_output=False)
            model[i].build_pred(model_file_name, config=config)
    elif model_type == "seq2seq":
        model = Img2InchiModel(config, output_dir='', vocab=my_vocab, need_output=False)
        model.build_pred(model_file_name, config=config)
    else:
        raise NotImplementedError("Unknown type of model!")

    sz = ceil(len(dir_list) / device_num)
    lock = threading.Lock()
    threads = [commit(i, model[i], my_vocab, dir_list[i * sz:(i + 1) * sz], batch_size, csv_path, lock) for i in range(device_num)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    main()
