import torch

from img2inchi import Img2InchiModel
from base import BaseModel
from model.Transformer import Img2SeqTransformer
from pkg.utils.utils import join, num_param

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


class Img2InchiTransformerModel(Img2InchiModel):
    def __init__(self, config, output_dir, vocab, need_output=True):
        super(Img2InchiTransformerModel, self).__init__(config, output_dir, vocab, need_output=need_output)

    def getModel(self):
        feature_size_1 = self._config.transformer["feature_size_1"]
        feature_size_2 = self._config.transformer["feature_size_2"]
        if feature_size_1 and feature_size_2:
            feature_size = (feature_size_1, feature_size_2)
        else:
            feature_size = None
        extractor_name = self._config.transformer["extractor_name"]
        tr_extractor = self._config.transformer["tr_extractor"]
        pretrain = self._config.transformer['pretrain']
        max_seq_len = self.max_len
        num_encoder_layers = self._config.transformer["num_encoder_layers"]
        num_decoder_layers = self._config.transformer["num_decoder_layers"]
        d_model = self._config.transformer["d_model"]
        nhead = self._config.transformer["nhead"]
        vocab_size = self.vocab_size
        if self._config.transformer["dim_feedforward"]:
            dim_feedforward = self._config.transformer["dim_feedforward"]
        else:
            dim_feedforward = 1024
        if self._config.transformer["dropout"]:
            dropout = self._config.transformer["dropout"]
        else:
            dropout = 0.1
        model = Img2SeqTransformer(feature_size, extractor_name, pretrain, max_seq_len, tr_extractor, num_encoder_layers,
                                   num_decoder_layers, d_model, nhead, vocab_size, dim_feedforward, dropout)
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
        if lr_scheduler == "AwesomeScheduler":
            d_model = self._config.transformer["d_model"]
            warmup_steps = self._config.warmup_steps
            lr_sc = lambda step: (d_model) ** (-0.5) * \
                                 (min((step + 1) ** (-0.5), (step + 1) * (warmup_steps ** (-1.5))))
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_sc)
            # if resume, restore lr
            if self.is_resume:
                scheduler.load_state_dict(self.old_model["scheduler"])
            return scheduler
        else:
            return super().getLearningRateScheduler(lr_scheduler)

    def save(self):
        """Saves model"""
        self.logger.info("- Saving model...")
        # save state as a dict
        if self.multi_gpu:
            model = self.model.module
        else:
            model = self.model
        optimizer = self.optimizer
        checking_point = {"net": model.state_dict(),
                          "optimizer": optimizer.state_dict(),
                          "epoch": self.now_epoch,
                          "scheduler": self.scheduler.state_dict()}
        if self._config.transformer["tr_extractor"]:
            torch.save(model.features_extractor.extractor.state_dict(), join(self._model_dir, 'pretrained_' + 
                        self._config.transformer['extractor_name'] + '.pth'))
        torch.save(checking_point, self._model_path)
        self._config.save(self._config_export_path)
        self.logger.info("- Saved model in {}".format(self._model_dir))