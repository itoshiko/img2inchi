import click

from data_gen import Img2SeqDataset
from img2inchi import Img2InchiModel
from img2inchi_transformer import Img2InchiTransformerModel
from pkg.utils.general import Config
from pkg.utils.vocab import vocab as vocabulary
from pkg.utils.LRScheduler import LRSchedule


@click.command()
@click.option('--data', default="./config/data_small.yaml",
              help='Path to data yaml config')
@click.option('--vocab', default="./config/vocab_small.yaml",
              help='Path to vocab yaml config')
@click.option('--training', default="./config/training_small.yaml",
              help='Path to training yaml config')
@click.option('--model', default="./config/model.yaml",
              help='Path to model yaml config')
@click.option('--output', default="./model weights/",
              help='Path to save trained model')
def main(data, vocab, training, model, output):
    # Load configs
    dir_output = output
    config = Config([data, vocab, training, model])
    config.save(dir_output)
    my_vocab = vocabulary(config.path_train_root)
    config.transformer["vocab_size"] = my_vocab.size

    # Load datasets
    train_set = Img2SeqDataset(root=config.path_train_root,
                               data_dir=config.path_train_data_dir,
                               img_dir=config.path_train_img_dir,
                               annotations_file=config.train_annotations_file)
    val_set = Img2SeqDataset(root=config.path_val_root,
                               data_dir=config.path_val_data_dir,
                               img_dir=config.path_val_img_dir,
                               annotations_file=config.val_annotations_file)

    # Define learning rate schedule
    n_batches_epoch = ((len(train_set) + config.batch_size - 1) // config.batch_size)
    lr_schedule = LRSchedule(lr_init=config.lr_init,
                             start_decay=config.start_decay * n_batches_epoch,
                             end_decay=config.end_decay * n_batches_epoch,
                             end_warm=config.end_warm * n_batches_epoch,
                             lr_warm=config.lr_warm,
                             lr_min=config.lr_min)
    # Build model and train
    if config.model_name == "seq2seq":
        model = Img2InchiModel(config, dir_output, my_vocab)
        model.build_train(config)
        model.train(config, train_set, val_set, lr_schedule)
    elif config.model_name == "transformer":
        model = Img2InchiTransformerModel(config, dir_output, my_vocab)
        model.build_train(config)
        model.train(config, train_set, val_set, lr_schedule)


if __name__ == "__main__":
    main()
