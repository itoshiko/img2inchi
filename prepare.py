import pkg.preprocess.data_preprocess as prc

BUILD_VOCAB = False
SPLIT_DATA_SET = False
PRC_IMG = True

VAL_SIZE = int(100e3)
TRAIN_SIZE = -1
root = "D:/Tsinghua/2021.2/Artificial_Intelligence/Final Project/img2inchi"
origin_dir = 'data/origin'
prcd_dir = 'data/prcd_data'

if __name__ == "__main__":
    data_set = prc.read_data_set(root=root, dir_name=origin_dir, file_name='train_labels.csv')
    if BUILD_VOCAB:
        prc.build_vocabulary(root=root, inchi_list=data_set['InChI'].values)
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
