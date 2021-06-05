from typing import Optional
import numpy as np
from pkg.utils.utils import join, read_img, save_img

import cv2


def pad_resize(img, config):
    h, w = img.shape
    pad_h, pad_v = 0, 0
    hw_ratio = (h / w) - (config["img_height"] / config["img_width"])
    if hw_ratio < 0:
        pad_h = int(abs(hw_ratio) * w / 2)
    else:
        wh_ratio = (w / h) - (config["img_width"] / config["img_height"])
        pad_v = int(abs(wh_ratio) * h // 2)
    img = np.pad(img, [(pad_h, pad_h), (pad_v, pad_v)], mode='constant', constant_values=255)
    img = cv2.resize(img, (config["img_width"], config["img_height"]), interpolation=cv2.INTER_LANCZOS4)
    return img


def preprocess(img: np.ndarray, config: dict):
    if config is None:
        config = {"img_height": 256, "img_width": 512, "threshold": 0, "phology": True}

    # rotate counter clockwise to get horizontal images
    h, w = img.shape
    if h > w:
        img = np.rot90(img)
    img = pad_resize(img, config)
    img = (img / img.max() * 255).astype(np.uint8)
    if config["threshold"] == -1:
        img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    elif config["threshold"] == 0:
        pass
    elif config["threshold"] > 0:
        img = cv2.threshold(img, config["threshold"], 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_not(img)
    if config["phology"]:
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    return img


def prc_img(img_id, source_root="train", target_root="prcd_data", config: Optional[dict] = None):
    img = read_img(root=source_root, img_id=img_id, mode='GRAY')
    img = preprocess(img, config)
    save_img(img=img, root=target_root, img_id=img_id)

    '''
    if DEBUG:
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    '''
