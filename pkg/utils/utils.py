import os
import numpy as np
import torch.nn as nn

try:
    import cv2
    has_cv2 = True
except:
    from PIL import Image
    has_cv2 = False


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
    if has_cv2:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = Image.open(img_path)
    return img

def num_param(model: nn.Module):
    num = 0
    for p in model.parameters():
        n = 1
        for d in p.shape:
            n *= d
        num += n
    return num