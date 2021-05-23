import os
import cv2
import torch
import click
import numpy as np

from pkg.utils.general import Config
from img2inchi import Img2InchiModel
from img2inchi_transformer import Img2InchiTransformerModel
from pkg.utils.vocab import vocab as vocabulary
from predict import pre_process


def predict_all(model, vocab, path, batch_size):
    """predict all for commit
    """
    img_list = []
    img_count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file[-3:] == "png" or file[-3:] == "jpg":
                _img_path = os.path.join(path, file)
                img = cv2.imread(_img_path, cv2.IMREAD_GRAYSCALE)
                _, img = pre_process(img)
                img = np.tile(img, (3, 1, 1))
                img_list.append(img)
                img_count += 1
    # split array
    arr_count = img_count // batch_size + 1
    split_array = np.array_split(np.array(img_list), arr_count)
    for arr in split_array:
        img_list = torch.from_numpy(arr).float()
        result = model.predict(img_list, mode="greedy")
        for j in range(len(result)):
            print(vocab.decode(result[j]))


@click.command()
@click.option('--model_path', default="./model weights/test_save",
              help='Path to trained model')
@click.option('--data_path', default="./images",
              help='Path to data to process')
@click.option('--batch_size', default=64,
              help='number of images to process at the same time')
def main(model_path, data_path, batch_size):
    model_file_name = model_path + '/model.cpkt'
    model_config = model_path + '/export_config.yaml'
    config = Config(model_config)
    my_vocab = vocabulary(root=config.path_train_root, vocab_dir=config.vocab_dir)
    model_type = config.model_name
    if model_type == "transformer":
        model = Img2InchiTransformerModel(config, output_dir='', vocab=my_vocab, need_output=False)
        model.build_pred(model_file_name, config=config)
        predict_all(model, my_vocab, data_path, batch_size)
    elif model_type == "seq2seq":
        model = Img2InchiModel(config, output_dir='', vocab=my_vocab, need_output=False)
        predict_all(model, my_vocab, data_path, batch_size)


if __name__ == '__main__':
    main()
