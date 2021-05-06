import os
import logging.config
import yaml

from shutil import copyfile


def setup_logger(default_path='./config/logging_config.yaml', default_level=logging.INFO):
    path = default_path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
        print('the input path doesn\'t exist')


def get_logger(default_path='./config/logging_config.yaml'):
    setup_logger(default_path)
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
            data = yaml.load(f)
            self.__dict__.update(data)

    def save(self, dir_name):
        init_dir(dir_name)
        if type(self.source) is list:
            for s in self.source:
                c = Config(s)
                c.save(dir_name)
        elif type(self.source) is dict:
            yaml.dump(self.source, dir_name)
        else:
            copyfile(self.source, dir_name + self.export_name)

    def show(self, fun=print):
        if type(self.source) is list:
            for s in self.source:
                c = Config(s)
                c.show(fun)
        elif type(self.source) is dict:
            fun(yaml.dump(self.source, ))
        else:
            with open(self.source) as f:
                fun(yaml.dump(yaml.load(f), ))
