# 数据预处理

## 数据文件组织

用于训练的图片和标签位于`data`文件夹下，包括：

* `origin`：存放原始数据（解压后）
* `prcd_data`：存放预处理完毕的数据
* `prcd_data_small`：存放预处理完毕的数据（小数据集）

## 数据的预处理

文件`prepare.py`用于对原始数据进行预处理，包括图像尺寸的调整和二值化、词表生成等，具体用法如下：

```bash
python prepare.py --root <project_root> --pre_config <prepare_config> --vocab_config <vocab_config>
# --root 项目的根目录，默认为 "./"
# --pre_config 数据生成与处理的配置文件的目录（根目录的相对路径），默认为 "./config/data_prepare_small.yaml"
# --vocab_config 词表配置文件目录（根目录的相对路径），默认为 "./config/vocab_small.yaml"
```

配置文件中各项配置作用如下表：

| 名称               | 作用                                   | 示例                           |
| ------------------ | -------------------------------------- | ------------------------------ |
| `origin_dir`       | 原始数据目录，与根目录的相对路径，下同 | `'data/origin'`                |
| `prcd_dir`         | 预处理后数据目录                       | `'data/prcd_data_small'`       |
| `train_labels`     | 原始数据集中标签csv文件名              | `'train_labels.csv'`           |
| `val_set_labels`   | 生成数据集中验证数据标签csv文件名      | `'small_val_set_labels.csv'`   |
| `train_set_labels` | 生成数据集中训练数据标签csv文件名      | `'small_train_set_labels.csv'` |
| `val_size`         | 验证集大小，整数                       | `20000`                        |
| `train_size`       | 训练集大小，-1为除验证集外的所有，整数 | `80000`                        |
| `build_vocab`      | 是否构建词表，布尔类型                 | `TRUE`                         |
| `split_data_set`   | 是否划分训练集和验证集，布尔类型       | `TRUE`                         |
| `prc_img`          | 是否对图像进行预处理，布尔类型         | `TRUE`                         |
| `img_width`        | 图像宽，整数                           | `512`                          |
| `img_height`       | 图像高，整数                           | `256`                          |
| `threshold`        | 二值化阈值，整数                       | `50`                           |
| `threads`          | 同时创建线程数，整数                   | `16`                           |

