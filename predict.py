import os
import cv2
import torch
import click
import numpy as np

from pkg.utils.general import Config
from img2inchi import Img2InchiModel
from img2inchi_transformer import Img2InchiTransformerModel
from pkg.utils.vocab import vocab as vocabulary
from pkg.preprocess.img_process import pad_resize


def pre_process(image):
    _config = Config('./config/data_prepare.yaml')
    # print(f"Worker-{self.process_id}: start processing images.")
    assert (_config is not None), "Can't get config file!"
    # rotate counter clockwise to get horizontal images
    h, w = image.shape
    if h > w:
        image = np.rot90(image)
    image = pad_resize(image, {"img_height": _config.img_height, "img_width": _config.img_width})
    image = (image / image.max() * 255).astype(np.uint8)
    if _config.threshold == -1 and _config.thresholding is True:
        image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    elif _config.threshold > 0 and _config.thresholding is True:
        image = cv2.threshold(image, _config.threshold, 255, cv2.THRESH_BINARY)
    else:
        pass
    img = cv2.bitwise_not(image)
    if _config.phology is True:
        image = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    return image


def interactive_shell(model, vocab):
    """Creates interactive shell to play with model
    """
    print("This is an interactive mode.To exit, enter 'exit'."
          "Enter a path to a fileinput> data/images_test/0.png")

    while True:
        # img_path = raw_input("input> ")# for python 2
        img_path = input("input> ")  # for python 3

        if img_path == "exit" or img_path == "q":
            break  # quit

        if (img_path[-3:] == "png" or img_path[-3:] == "jpg") and (os.path.isfile(img_path)):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = pre_process(img)
            img = torch.from_numpy(img).float()
            img = img.repeat(1, 3, 1, 1)
            result = model.predict(img, mode="beam")
            seq = vocab.decode(result[0])
            print(seq)
        if os.path.isdir(img_path):
            img_list = []
            for root, dirs, files in os.walk(img_path):
                for file in files:
                    _img_path = os.path.join(img_path, file)
                    img = cv2.imread(_img_path, cv2.IMREAD_GRAYSCALE)
                    img = pre_process(img)
                    img = np.tile(img, (3, 1, 1))
                    img_list.append(img)
            img_list = torch.Tensor(img_list).float()
            result = model.predict(img_list, mode="greedy")
            for i in range(result.shape[0]):
                print(vocab.decode(result[i]))


@click.command()
@click.option('--model_path', default="",
              help='Path to trained model')
@click.option('--instance', default="",
              help='Name of your model file')
def main(model_path, instance):
    # restore config and model_read
    # TODO cpkt -> ckpt
    model_file_name = model_path + '/' + instance + '.ckpt'
    model_config = model_path + '/export_config.yaml'
    config = Config(model_config)
    my_vocab = vocabulary(root=config.path_train_root, vocab_dir=config.vocab_dir)
    model_type = config.model_name
    if model_type == "transformer":
        model = Img2InchiTransformerModel(config, output_dir='', vocab=my_vocab, need_output=False)
        model.build_pred(model_file_name, config=config)
        interactive_shell(model, my_vocab)
    elif model_type == "seq2seq":
        model = Img2InchiModel(config, output_dir='', vocab=my_vocab, need_output=False)
        model.build_pred(model_file_name, config=config)
        interactive_shell(model, my_vocab)


if __name__ == '__main__':
    main()
