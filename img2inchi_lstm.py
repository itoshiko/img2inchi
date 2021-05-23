import torch

from base import BaseModel
from model.Img2Seq import Img2Seq
from pkg.utils.utils import num_param
from img2inchi import Img2InchiModel

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


class Img2InchiLstmModel(Img2InchiModel):
    def __init__(self, config, output_dir, vocab, need_output=True):
        super(Img2InchiLstmModel, self).__init__(config, output_dir, vocab, need_output=need_output)

    def getModel(self):
        img_w = self._config.img2seq['img_w']
        img_h = self._config.img2seq['img_h']
        vocab_size = self._config.vocab_size
        dim_encoder = self._config.img2seq['dim_encoder']
        dim_decoder = self._config.img2seq['dim_decoder']
        dim_attention = self._config.img2seq['dim_attention']
        dim_embed = self._config.img2seq['dim_embed']
        if self._config.img2seq['dropout']:
            dropout = self._config.img2seq['dropout']
        else:
            dropout = 0.5
        model = Img2Seq(img_w, img_h, vocab_size, dim_encoder, dim_decoder, dim_attention, dim_embed, dropout=dropout)
        self.model = model
        print(f'The number of parameters: {num_param(model)}')
        return model

    def getOptimizer(self, lr_method="adam", lr=1e-3):
        if lr_method == 'adam':
            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        elif lr_method == 'adamax':
            optimizer = torch.optim.Adamax(params=self.model.parameters(), lr=lr)
        elif lr_method == 'sgd':
            optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError("Unknown Optimizer {}".format(lr_method))
        return optimizer

    def getLearningRateScheduler(self, lr_scheduler="CosineAnnealingLR"):
        return super().getLearningRateScheduler(lr_scheduler)
