import numpy as np
import cv2
from ..utils.utils import join


def pad_resize(img, config):
    h, w = img.shape
    pad_h, pad_v = 0, 0
    hw_ratio = (h / w) - (config["img_height"] / config["img_width"])
    if hw_ratio < 0:
        pad_h = int(abs(hw_ratio) * w / 2)
    else:
        wh_ratio = (w / h) - (config["img_width"] / config["img_height"])
        pad_v = int(abs(wh_ratio) * h // 2)
    img = np.pad(img, [(pad_h, pad_h), (pad_v, pad_v)], mode='constant')
    img = cv2.resize(img, (config["img_width"], config["img_height"]), interpolation=cv2.INTER_NEAREST)
    return img


def prc_img(img_id, source_root="train", target_root="prcd_data", config=None):
    if config is None:
        config = {"img_height": 256, "img_width": 512, "threshold": 50}
    source_file_path = join(source_root, img_id[0], img_id[1], img_id[2], f'{img_id}.png')
    target_file_path = join(target_root, img_id[0], img_id[1], img_id[2], f'{img_id}.png')
    img = 255 - cv2.imread(source_file_path, cv2.IMREAD_GRAYSCALE)

    # rotate counter clockwise to get horizontal images
    h, w = img.shape
    if h > w:
        img = np.rot90(img)
    img = pad_resize(img, config)
    img = (img / img.max() * 255).astype(np.uint8)
    img[np.where(img > config["threshold"])] = 255
    img[np.where(img <= config["threshold"])] = 0
    cv2.imwrite(target_file_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    '''
    if DEBUG:
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    '''
