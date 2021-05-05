import os
import numpy as np
import torch
import cv2


def join(path, *subdirs):
    for dir in subdirs:
        path = os.path.join(path, dir)
    return path

def create_dirs(path, *subdirs):
    if len(subdirs) > 0:
        path = join(path, *subdirs)
    if not os.path.exists(path):
        os.makedirs(path)

def get_img_path(img_id, path):
    return join(path, img_id[0], img_id[1], img_id[2], f'{img_id}.png')

def read_img(img_id, root):
    '''
    read image by cv2
    '''
    img_path = get_img_path(img_id, root)
    # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path)
    return img

def one_hot(seqs, vocab_size):
    d1, d2 = seqs.shape
    seqs = seqs.unsqueeze(-1)
    one_hot = torch.zeros(d1, d2, vocab_size)
    one_hot.scatter_(dim=2, index=seqs.long(), src=torch.ones(d1, d2, vocab_size))
    return one_hot


