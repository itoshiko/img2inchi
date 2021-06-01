import os
import cv2
from numpy.core.fromnumeric import shape
import torch
import click
import numpy as np

from pkg.utils.general import Config
from img2inchi_lstm import Img2InchiLstmModel
from img2inchi_transformer import Img2InchiTransformerModel
from pkg.utils.vocab import vocab as vocabulary
from pkg.preprocess.img_process import preprocess


def pre_process(image):
    _config = Config('./config/data_prepare.yaml')
    # print(f"Worker-{self.process_id}: start processing images.")
    assert (_config is not None), "Can't get config file!"
    # rotate counter clockwise to get horizontal images
    return preprocess(img=image, config=_config.__dict__)


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
            raw_img = img
            img = torch.from_numpy(img).float()
            img = img.repeat(1, 3, 1, 1)
            results = model.predict(img, mode="beam")
            attn = model.get_attention(img, results)[0, 0].cpu().numpy()
            nhead, lenth, h, w = attn.shape
            assert nhead == 8
            attn = np.reshape(np.transpose(attn, (1, 0, 2, 3)), newshape=(lenth, 4, 2, h, w))
            attn = np.reshape(np.transpose(attn, (0, 1, 3, 2, 4)), newshape=(lenth, 4 * h, 2 * w))
            imgh, imgw = raw_img.shape
            imgh = int(imgh / 1.6)
            imgw = int(imgw / 1.6)
            raw_img = cv2.resize(raw_img, (imgw, imgh), interpolation=cv2.INTER_LANCZOS4)
            print(raw_img.shape, h, w)
            raw_img = np.reshape(np.transpose(np.tile(raw_img, reps=(4, 2, 1, 1)), (0, 2, 1, 3)), newshape=(4 * imgh, 2 * imgw))
            attn = np.transpose(attn / np.max(attn, axis=0, keepdims=True), (1, 2, 0))
            attn = cv2.resize(attn, (4 * imgh, 2 * imgw), interpolation=cv2.INTER_LANCZOS4)
            cv2.namedWindow("Attention")
            for i in range(lenth):
                x = cv2.addWeighted(raw_img / 255, 0.5, attn[:, :, i], 0.5, 0, dtype=cv2.CV_32FC1)
                #x = np.expand_dims(x, axis=-1)
                print(vocab(int(results[0, i + 1].item())))
                cv2.imshow("Attention", x)
                cv2.waitKey(0)
            cv2.destroyAllWindows()
            results = vocab.decode(results)
            for r in results:
                print(r)
        elif os.path.isdir(img_path):
            img_list = []
            for root, dirs, files in os.walk(img_path):
                for file in files:
                    _img_path = os.path.join(img_path, file)
                    img = cv2.imread(_img_path, cv2.IMREAD_GRAYSCALE)
                    img = pre_process(img)
                    img = np.tile(img, (3, 1, 1))
                    img_list.append(img)
            img_list = torch.Tensor(img_list).float()
            result = vocab.decode(model.predict(img_list, mode="greedy"))
            for i in range(result.shape[0]):
                print(result[i])


@click.command()
@click.option('--model_path', default="",
              help='Path to trained model')
@click.option('--instance', default="",
              help='Name of your model file')
def main(model_path, instance):
    # restore config and model_read
    model_file_name = model_path + '/' + instance + '.ckpt'
    model_config = model_path + '/export_config.yaml'
    config = Config(model_config)
    my_vocab = vocabulary(root=config.vocab_root, vocab_dir=config.vocab_dir)
    model_type = config.model_name
    if model_type == "transformer":
        model = Img2InchiTransformerModel(config, output_dir='', vocab=my_vocab, need_output=False)
        model.build_pred(model_file_name, config=config)
        interactive_shell(model, my_vocab)
    elif model_type == "seq2seq":
        model = Img2InchiLstmModel(config, output_dir='', vocab=my_vocab, need_output=False)
        model.build_pred(model_file_name, config=config)
        interactive_shell(model, my_vocab)


if __name__ == '__main__':
    main()
