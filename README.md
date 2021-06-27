# img2inchi

本项目可以将分子骨架式转换为InChI表达式。

<img src="./figures/1.png" style="zoom:60%;" />

## 安装依赖

## 配置文件
将config_templates文件夹复制一份为config文件夹，所有的配置将默认从这个文件夹读取。
具体的配置稍后说明。

## 准备训练数据

### 数据下载
在Kaggle上直接下载数据到/data下，解压到data/origin(推荐)。
https://www.kaggle.com/c/bms-molecular-translation

确认解压后的数据文件夹(data/origin)包括：
* test/
* train/
* extra_approved_InChIs.csv
* train_labels.csv


### 数据预处理
这部分的详细说明参考data/README.md
#### 配置参数
根据生成数据集大小，我们有两个预置的模板：
* data_prepare_small.yaml: 小数据集
* data_prepare.yaml: 全数据集

可以根据自己的需求直接调用或者在此基础上修改配置。
具体的配置含义请参考data/README.md

#### 预处理
在项目根目录下运行：
```bash
python prepare.py --root <project_root> --pre_config <prepare_config>
# --root 项目的根目录，默认为 "./"
# --vocab_config 词表配置文件目录（根目录的相对路径），默认为 "./config/vocab.yaml"
```

#### 配置数据集配置文件
根据生成数据集大小，我们有两个预置的模板：
* data_small.yaml: 小数据集
* data.yaml: 全数据集

可以根据自己的需求直接调用或者在此基础上修改配置。  
配置文件中各项配置作用如下表：

| 名称                      | 作用                                       | 示例                           |
| ------------------------- | ----------------------------------------- | ------------------------------ |
| `path_train_root`         | 训练集根目录。                              | `'./'`                         |
| `path_train_data_dir`     | 训练集目录，相对根目录的相对路径。           | `'data/prcd_data_small/'`      |
| `path_train_img_dir`      | 训练集数据文件夹，相对训练集目录的相对路径。  | `'train'`                      |
| `train_annotations_file`  | 训练集数据索引文件，相对训练集目录的相对路径。| `'small_train_set_labels.csv'` |
| `path_val_root`           | 验证集根目录。                              | `'./'`                         |
| `path_val_data_dir`       | 验证集目录，相对根目录的相对路径。           | `'data/prcd_data_small/'`      |
| `path_val_img_dir`        | 验证集数据文件夹，相对验证集目录的相对路径。  | `'validate'`                   |
| `val_annotations_file`    | 验证集数据索引文件，相对验证集目录的相对路径。| `'small_val_set_labels.csv'`   |
| `dataloader_num_workers`  | DataLoader加载数据时的进程数目。             | `2`                           |

## 训练
### 训练中超参
在模型配置文件中，有一些训练相关的参数，这些参数的格式是transformer和lstm模型通用的。

| 名称                         | 作用                                                      | 示例                           |
| ---------------------------- | -------------------------------------------------------- | ------------------------------ |
| `model_name`                 | 模型名称，`'transformer'`或`'lstm'`。字符串               | `'transformer'`                |
| `max_seq_len`                | 最大序列长度，请勿修改。整数型                             | `300`                          |
| `batch_size`                 | Batch size，请根据gpu显存大小适当调整。整数型              | `50`                           |
| `eval_batch_size`            | 评估阶段的Batch size，一般可以batch_size大很多。整数型     | `800`                          |
| `lr_method`                  | 优化器，可以为`'adam', 'adamax', 'sgd'`。字符串           | `'adam'`                       |
| `lr_scheduler`               | 学习率策略，可以为`'CosineAnnealingLR', 'AwesomeScheduler'`。字符串| `'AwesomeScheduler'`   |
| `device`                     | 训练时的设备，`'cpu', 'cuda'`等。字符串                   | `'cuda:0'`                     |
| `multi_gpu`                  | 是否采用多gpu训练，无多个gpu时将忽略该参数。布尔型          | `True`                         |
| `criterion_method`           | Loss函数，此问题必须选`'CrossEntropyLoss'`。字符串         | `'CrossEntropyLoss'`          |
| `n_epochs`                   | 训练的轮数。整数型                                        | `20`                           |
| `SCST_predict_mode`          | SCST中推理的模式，可以为`'greedy', 'beam'`。字符串        | `'greedy'`                     |
| `SCST_lr`                    | SCST学习率，一般较小。浮点型                              | `0.000001`                     |
| `gradient_accumulate_num`    | 梯度累积次数，将梯度累积次数乘以batch_size得到实际Batch大小，一般达到200即可。整数型| `4`     |
| `beam_width`                 | 束搜索的宽度。整数型                                      | `5`                            |

### 训练transformer

#### 准备预训练模型
先运行model_weights/load resnet.py文件，下载PyTorch中的预训练resnet模型。  
训练中需要的预训练特征提取器可以自行训练，也可以使用我们的预训练模型。  
如果要自行训练预训练模型，请先修改config/pretrain_config.yaml文件中的参数，详细含义参考本节给出的表格。  
一般情况下，transformer的参数请不要修改。  
然后在项目根目录下运行：
```bash
python train.py --model_name transformer --instance pretrain --data ./config/data.yaml --model ./config/pretrain_config.yaml
```

#### 配置模型参数
预置模板为transformer_config.yaml  
可以根据自己的需求直接调用或者在此基础上修改配置。  
配置文件中各项配置作用如下表：

| 名称                            | 作用                                                      | 示例                           |
| ------------------------------- | -------------------------------------------------------- | ------------------------------ |
| `transformer.feature_size_1`    | 输出的特征图高度，为空则表示用默认尺寸。整数型|None         | `None`                         |
| `transformer.feature_size_2`    | 输出的特征图宽度，为空则表示用默认尺寸。整数型|None         | `None`                         |
| `transformer.extractor_name`    | 特征提取器的名称，`'resnet34'`或`'resnet101'`。字符串      | `'resnet34'`                   |
| `transformer.pretrain`          | 预训练特征提取器的路径，`''`表示读取默认的预训练模型，`'none'`表示随机初始化参数。字符串| `'./model_weights/pretrained_resnet34.pth'`|
| `transformer.tr_extractor`      | 是否训练特征提取器。布尔型                                 | `FALSE`                        |
| `transformer.num_encoder_layers`| Encoder层数。整数型                                       | `10`                           |
| `transformer.num_decoder_layers`| Decoder层数。整数型                                       | `20`                           |
| `transformer.d_model`           | 模型维数或宽度。整数型                                     | `512`                          |
| `transformer.nhead`             | Attention中的头数。整数型                                 | `8`                            |
| `transformer.dim_feedforward`   | 逐点前馈的中间层维数。整数型                               | `1024`                         |
| `transformer.dropout`           | Drop out概率。浮点型                                      | `0.1`                          |
| `model_name`                    | 模型名称，`'transformer'`。字符串                         | `'transformer'`                |
| `warmup_steps`                  | Warm up步数，建议不要修改。整数型                          | `2000`                         |
| `lr_init`                       | 学习率系数，乘到学习率上。浮点型                           | `0.1`                          |
| `lr_max`                        | 最大学习率。浮点型                                        | `0.0006`                        |
| `lr_scheduler`                  | 学习率策略，`'AwesomeScheduler'`，字符串                   | `'AwesomeScheduler'`           |

其余训练中参数参考之前的表格设置。

#### 正式训练
在项目根目录下运行：
```bash
python train.py [--model_name transformer] [--instance <instance_name>] [--data <data_config_path>] [--vocab][--model][--scst][--output]
# --model_name 模型名称，默认为 transformer。
# --instance 实例名，默认为 test
# --data 数据配置文件路径，默认为 "./config/data_small.yaml"
# --vocab 词表配置文件路径，默认为 "./config/vocab.yaml"
# --model 模型配置文件路径，默认为 ""，表示根据模型名称从./config/中找
# --scst  是否进行SCST训练，默认为 False
# --output 模型输出与保存的根目录，默认为 "./model_weights"。模型输出及ckpt将保存至<output>/<instance>下
例：
python train.py --instance my_train --data ./config/data.yaml
```
如果instance已经存在，并且找到有export.yaml和model.ckpt文件，将会在check point的基础上恢复训练。  
注意，此时模型配置读取的是instance中导出的配置，但数据配置采用的是命令行中指定的配置。

### 训练seq2seq
未维护，可能无法运行
### 使用SCST
未维护，可能无法运行
## GUI

## 参考文献