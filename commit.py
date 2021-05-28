import os
import cv2
import csv
import torch
import click
import numpy as np

from pkg.utils.general import Config
from img2inchi import Img2InchiModel
from img2inchi_transformer import Img2InchiTransformerModel
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
                _, img = pre_process(img)
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
        seq = seq + vocab.decode(model.predict(img_list, mode="beam"))
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
    model_file_name = model_path + '/model.cpkt'
    model_config = model_path + '/export_config.yaml'
    config = Config(model_config)
    my_vocab = vocabulary(root=config.path_train_root, vocab_dir=config.vocab_dir)
    model_type = config.model_name
    dir_list = walk_all_dir(data_path)
    seq_list = []
    name_list = []
    if model_type == "transformer":
        model = Img2InchiTransformerModel(config, output_dir='', vocab=my_vocab, need_output=False)
        model.build_pred(model_file_name, config=config)
    elif model_type == "seq2seq":
        model = Img2InchiModel(config, output_dir='', vocab=my_vocab, need_output=False)
        model.build_pred(model_file_name, config=config)
    else:
        raise NotImplementedError("Unknown type of model!")

    i = 0

    for directory in dir_list:
        small_seq_list, small_name_list = predict_all(model, my_vocab, directory, batch_size)
        seq_list = seq_list + small_seq_list
        name_list = name_list + small_name_list
        if len(seq_list) > 5000:
            write_csv(seq_list, name_list, os.path.join(data_path, csv_filename))
            seq_list.clear()
            name_list.clear()
        i += 1
        print('Processed {} folders'.format(i))
    write_csv(seq_list, name_list, os.path.join(data_path, csv_filename))


if __name__ == '__main__':
    main()
