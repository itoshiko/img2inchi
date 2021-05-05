import numpy as np
import cv2
from ..utils.utils import join

data_root = "D:/Tsinghua/2021.2/Artificial_Intelligence/Final Project/data"

IMG_WIDTH = 512
IMG_HEIGHT = 256
THRESHOLD = 50

def pad_resize(img):
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

def prc_img(img_id, source_folder="train", target_folder="prcd_data"):
    source_file_path = join(data_root, source_folder, img_id[0], img_id[1], img_id[2], f'{img_id}.png')
    target_file_path = join(data_root, target_folder, img_id[0], img_id[1], img_id[2], f'{img_id}.png')
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
    '''
    if DEBUG:
        cv2.namedWindow("Image")
        cv2.imshow("Image", img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    '''
