from os import makedirs
from typing import TypeVar, Union
from numpy import array, ndarray

try:
    import cv2

    has_cv2 = True
except:
    from PIL import Image

    has_cv2 = False


def _join(path1: str, path2: str):
    if path1 == '':
        return path2
    if path2 == '':
        return path1
    if path1[-1] == '/':
        p1 = path1[:-1]
    else:
        p1 = path1
    if path2[0] == '/':
        p2 = path2[1:]
    else:
        p2 = path2
    return p1 + '/' + p2


def join(*paths):
    p = ''
    for path in paths:
        p = _join(p, path)
    return p


def create_dirs(root, *subdirs):
    path = join(root, *subdirs)
    makedirs(path, exist_ok=True)


def get_img_path(root, img_id, format='png'):
    return join(root, img_id[0], img_id[1], img_id[2], f'{img_id}.{format}')


def read_img(root: str, img_id: str, mode: str = 'GRAY') -> ndarray:
    '''
    read image by cv2/PIL
    :param root: the image file's root
    :param img_id: the id of image
    :param mode: RGB or GRAY
    :return: np.ndarray
    '''
    img_path = get_img_path(root=root, img_id=img_id)
    if has_cv2:
        if mode == 'RGB':
            img = cv2.imread(img_path)[..., ::-1]
        elif mode == 'GRAY':
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            raise NotImplementedError(f'Unkown mode: {mode}. Should be RGB or GRAY')
    else:
        img = Image.open(img_path)
        if mode == 'RGB':
            img = img.convert('RGB')
        elif mode == 'GRAY':
            pass
        else:
            raise NotImplementedError(f'Unkown mode: {mode}. Should be RGB or GRAY')
        img = array(img)
    return img


def save_img(img: ndarray, root: str, img_id: str):
    '''
    save image by cv2
    :param img: the np.ndarray typed img that to be saved
    :param root: the image file's root
    :param img_id: the id of image
    :return: None
    '''
    img_path = get_img_path(root=root, img_id=img_id)
    cv2.imwrite(img_path, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


def num_param(model):
    num = 0
    for p in model.parameters():
        n = 1
        for d in p.shape:
            n *= d
        num += n
    return num


T = TypeVar('T')


def split_list(l: Union['list[T]', 'tuple[T]'], d: int) -> 'list[list[T]]':
    return [l[i:i + d] for i in range(0, len(l), d)]


def flatten_list(l: 'list[list[T]]') -> 'list[T]':
    return sum(l, [])
