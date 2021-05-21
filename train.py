import click

from data_gen import Img2SeqDataset
from img2inchi import Img2InchiModel
from img2inchi_transformer import Img2InchiTransformerModel
from pkg.utils.general import Config
from pkg.utils.vocab import vocab as vocabulary
from pkg.utils.LRScheduler import LRSchedule


@click.command()
@click.option('--model_name', default="transformer",
              help='Model to train')
@click.option('--instance', default="test",
              help='Name of instance to train')
@click.option('--data', default="./config/data_small.yaml",
              help='Path to data yaml config')
@click.option('--vocab', default="./config/vocab.yaml",
              help='Path to vocab yaml config')
@click.option('--model', default="",
              help='Path to model yaml config')
@click.option('--output', default="./model weights",
              help='Path to save trained model')
def main(model_name, instance, data, vocab, model, output):
    # Load configs
    dir_output = output
    if model == "":
        model = './config/' + model_name + '_config.yaml'
    config = Config([data, vocab, model])
    my_vocab = vocabulary(root=config.path_train_root, vocab_dir=config.vocab_dir)
    config.vocab_size = my_vocab.size
    config.model_name = model_name
    config.instance = instance

    # Load datasets
    train_set = Img2SeqDataset(root=config.path_train_root,
                               data_dir=config.path_train_data_dir,
                               img_dir=config.path_train_img_dir,
                               annotations_file=config.train_annotations_file,
                               vocab=my_vocab)
    val_set = Img2SeqDataset(root=config.path_val_root,
                               data_dir=config.path_val_data_dir,
                               img_dir=config.path_val_img_dir,
                               annotations_file=config.val_annotations_file,
                               vocab=my_vocab)

    # Build model and train
    if model_name == "seq2seq":
        model = Img2InchiModel(config, dir_output, my_vocab)
        model.build_train(config)
        model.train(config, train_set, val_set)
    elif model_name == "transformer":
        model = Img2InchiTransformerModel(config, dir_output, my_vocab)
        model.build_train(config)
        model.train(config, train_set, val_set)


if __name__ == "__main__":
    main()
