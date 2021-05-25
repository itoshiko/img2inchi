import os
import logging.config
import yaml

from pkg.utils.utils import join


def setup_logger(log_path="./logs", default_path='./config/logging_config.yaml', default_level=logging.INFO):
    path = default_path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        f = config['handlers']['info_file_handler']['filename']
        config['handlers']['info_file_handler']['filename'] = join(log_path, f)
        f = config['handlers']['error_file_handler']['filename']
        config['handlers']['error_file_handler']['filename'] = join(log_path, f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        print('the input path doesn\'t exist')


def get_logger(log_path="./logs", default_path='./config/logging_config.yaml'):
    setup_logger(log_path=log_path, default_path=default_path)
    logging.info('Logger initialized', exc_info=True)
    return logging.getLogger()


def init_dir(dir_name):
    if dir_name is not None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


class Config:
    """Class that loads hyper parameters from json file into attributes"""

    def __init__(self, source, export_name="export_config.yaml"):
        """
        Args:
            source: path to json file or dict
        """
        self.source = source
        self.export_name = export_name

        if type(source) is dict:
            self.__dict__.update(source)
        elif type(source) is list:
            for s in source:
                self.load_yaml(s)
        else:
            self.load_yaml(source)

    def load_yaml(self, source):
        with open(source) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            self.__dict__.update(data)

    def save(self, dir_name):
        init_dir(dir_name)
        file_name = dir_name + '/' + self.export_name
        f = open(file_name, "w")
        yaml.dump(self.__dict__, f)

    def show(self, fun=print):
        if type(self.source) is list:
            for s in self.source:
                c = Config(s)
                c.show(fun)
        elif type(self.source) is dict:
            fun(yaml.dump(self.source, ))
        else:
            with open(self.source) as f:
                fun(yaml.dump(yaml.load(f, Loader=yaml.FullLoader), ))
