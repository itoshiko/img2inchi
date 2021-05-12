import click

import pkg.preprocess.data_preprocess as prc

from pkg.utils.vocab import build_vocabulary

from pkg.utils.general import Config


@click.command()
@click.option('--root', default="./",
              help="Path to your project root")
@click.option('--pre_config', default="./config/data_prepare.yaml",
              help='Path to config yaml for data preparation')
@click.option('--vocab_config', default="./config/vocab_small.yaml",
              help='Path to vocab yaml config')
def main(root, pre_config, vocab_config):
    data_config = Config([pre_config, vocab_config])
    origin_dir = data_config.origin_dir
    prcd_dir = data_config.prcd_dir
    vocab_dir = data_config.vocab_dir
    VAL_SIZE = data_config.val_size
    TRAIN_SIZE = data_config.train_size
    SPLIT_DATA_SET = data_config.split_data_set
    BUILD_VOCAB = data_config.build_vocab
    PRC_IMG = data_config.prc_img
    data_set = prc.read_data_set(root=root, dir_name=origin_dir, file_name='train_labels.csv')
    if BUILD_VOCAB:
        build_vocabulary(root=root, vocab_dir=vocab_dir, inchi_list=data_set['InChI'].values)
    if SPLIT_DATA_SET:
        val_set, train_set = prc.train_val_split(root=root, dir_name=prcd_dir, data_set=data_set,
                                                 train_size=TRAIN_SIZE, val_size=VAL_SIZE)
    else:
        val_set = prc.read_data_set(root=root, dir_name=prcd_dir, file_name='val_set_labels.csv')
        train_set = prc.read_data_set(root=root, dir_name=prcd_dir, file_name='train_set_labels.csv')
    print(train_set.info())
    print(val_set.info())
    if PRC_IMG:
        train_img_list = train_set['image_id'].values
        val_img_list = val_set['image_id'].values
        prc.prc_imgs(root=root, origin_dir_name=origin_dir, prcd_dir_name=prcd_dir,
                     name='train', img_list=train_img_list, num_threads=8)
        prc.prc_imgs(root=root, origin_dir_name=origin_dir, prcd_dir_name=prcd_dir,
                     name='validate', img_list=val_img_list, num_threads=8)


if __name__ == "__main__":
    main()