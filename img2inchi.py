import os
import sys
import time
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from base import BaseModel
from NetModel_Img2Seq import Img2Seq
from NetModel_Transformer import Img2SeqTransformer
from data_gen import Img2SeqDataset
from data_gen import get_dataLoader


class Img2InchiModel(BaseModel):
    def __init__(self, config, output_dir, vocab):
        super(Img2InchiModel, self).__init__(config, output_dir)
        self._vocab = vocab

    def getModel(self, model_name="Img2Seq"):
        if model_name == "Img2Seq":
            img_w = self._config.img2seq['img_w']
            img_h = self._config.img2seq['img_h']
            vocab_size = self._config.img2seq['vocab_size']
            dim_encoder = self._config.img2seq['dim_encoder']
            dim_decoder = self._config.img2seq['dim_decoder']
            dim_attention = self._config.img2seq['dim_attention']
            dim_embed = self._config.img2seq['dim_embed']
            if self._config.img2seq['dropout']:
                dropout = self._config.img2seq['dropout']
            else:
                dropout = 0.5
            model = Img2Seq(img_w, img_h, vocab_size, dim_encoder, dim_decoder, dim_attention, dim_embed, dropout=0.5)
            self.model = model
            return model
        if model_name == "Img2SeqTransformer":
            feature_size = (self._config.transformer['feature_size_1'], self._config.transformer['feature_size_1'])
            extractor_name = self._config.transformer['extractor_name']
            max_seq_len = self._config.transformer['max_seq_len']
            tr_extractor = self._config.transformer['tr_extractor']
            num_encoder_layers = self._config.transformer['num_encoder_layers']
            num_decoder_layers = self._config.transformer['num_decoder_layers']
            d_model = self._config.transformer['d_model']
            nhead = self._config.transformer['nhead']
            vocab_size = self._config.transformer['vocab_size']
            if self._config.transformer['dim_feedforward']:
                dim_feedforward = self._config.transformer['dim_feedforward']
            else:
                dim_feedforward = 1024
            if self._config.transformer['dropout']:
                dropout = self._config.transformer['dropout']
            else:
                dropout = 1024
            model = Img2SeqTransformer(feature_size, extractor_name, max_seq_len,
                                       tr_extractor, num_encoder_layers, num_decoder_layers,
                                       d_model, nhead, vocab_size, dim_feedforward, dropout)
            self.model = model
            return model

    def getOptimizer(self, lr_method="adam", lr=1e-3):
        if lr_method == 'adam':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        elif lr_method == 'adamax':
            self.optimizer = torch.optim.Adamax(params=self.model.parameters(), lr=lr)
        elif lr_method == 'sgd':
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError("Unknown Optimizer {}".format(lr_method))
        return super().getOptimizer(lr_method=lr_method, lr=lr)

    def _run_train_epoch(self, config, train_set, val_set, epoch, lr_schedule):
        """Performs an epoch of training
                Args:
                    config: Config instance
                    train_set: Dataset instance
                    val_set: Dataset instance
                    epoch: (int) id of the epoch, starting at 0
                    lr_schedule: LRSchedule instance that takes care of learning proc
                Returns:
                    score: (float) model will select weights that achieve the highest score
                """
        # logging
        batch_size = config.batch_size
        nbatches = (len(train_set) + batch_size - 1) // batch_size
        self.model.train()
        root = self._config
        train_loader = get_dataLoader(Img2SeqDataset(train_set),
                                                   batch_size=batch_size,
                                                   shuffle=True, num_workers=3, pin_memory=True)

        # for i, (img, formula) in enumerate(train_loader):
        for i, (img, formula) in enumerate(minibatches(train_set, batch_size)):
            img = pad_batch_images_2(img)
            img = torch.FloatTensor(img)  # (N, W, H, C)
            formula, formula_length = pad_batch_formulas(formula, self._vocab.id_pad, self._vocab.id_end)
            img = img.permute(0, 3, 1, 2)  # (N, C, W, H)
            formula = torch.LongTensor(formula)  # (N,)

            loss_eval = self.getLoss(img, formula=formula, lr=lr_schedule.lr, dropout=config.dropout, training=True)
            prog.update(i + 1, [("loss", loss_eval), ("lr", lr_schedule.lr)])

            # update learning rate
            lr_schedule.update(batch_no=epoch * nbatches + i)

        self.logger.info("- Training: {}".format(prog.info))
        self.logger.info("- Config: (before evaluate, we need to see config)")
        config.show(fun=self.logger.info)

        # evaluation
        config_eval = Config({"dir_answers": self._dir_output + "formulas_val/", "batch_size": config.batch_size})
        scores = self.evaluate(config_eval, val_set)
        score = scores["perplexity"]
        lr_schedule.update(score=score)

        return score
