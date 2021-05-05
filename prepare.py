import pkg.preprocess.data_preprocess as prc

BUILD_VOCAB = False
SPLIT_DATA_SET = False
PRC_IMG = False

if __name__ == "__main__":
    data_set = prc.read_data_set('train_labels.csv')
    if BUILD_VOCAB:
        prc.build_vocabulary(data_set)
    if SPLIT_DATA_SET:
        val_set, train_set = prc.train_val_split(data_set)
    print(train_set.info())
    print(val_set.info())
    if PRC_IMG:
        prc.prc_imgs(train_set, 'train')
        prc.prc_imgs(val_set, 'validate')
