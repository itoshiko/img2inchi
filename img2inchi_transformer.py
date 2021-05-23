import torch
from torch.optim import optimizer
from math import ceil

from torch.tensor import Tensor

import model.SelfCritical as SelfCritical

from base import BaseModel
from data_gen import get_dataLoader
from model.Transformer import Img2SeqTransformer
from pkg.utils.ProgBar import ProgressBar
from pkg.utils.BeamSearchTransformer import BeamSearchTransformer
from pkg.utils.utils import flatten_list, num_param

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


class Img2InchiTransformerModel(BaseModel):
    def __init__(self, config, output_dir, vocab, need_output=True):
        super(Img2InchiTransformerModel, self).__init__(config, output_dir, need_output=need_output)
        self._vocab = vocab
        self._device = config.device
        if self._device is None:
            self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.max_len = config.max_seq_len

    def build_train(self, config=None):
        res = super().build_train(config=config)
        self.beam_search = BeamSearchTransformer(tf_model=self.model, device=self._device, beam_width=config.beam_width,
                                                 topk=1, max_len=self.max_len, max_batch=config.batch_size)
        return res

    def build_pred(self, model_path, config=None):
        res = super().build_pred(model_path=model_path, config=config)
        self.beam_search = BeamSearchTransformer(tf_model=self.model, device=self._device, beam_width=config.beam_width,
                                                 topk=1, max_len=self.max_len, max_batch=config.batch_size)
        return res

    def getModel(self):
        feature_size_1 = self._config.transformer["feature_size_1"]
        feature_size_2 = self._config.transformer["feature_size_2"]
        if feature_size_1 and feature_size_2:
            feature_size = (feature_size_1, feature_size_2)
        else:
            feature_size = None
        extractor_name = self._config.transformer["extractor_name"]
        max_seq_len = self.max_len
        tr_extractor = self._config.transformer["tr_extractor"]
        num_encoder_layers = self._config.transformer["num_encoder_layers"]
        num_decoder_layers = self._config.transformer["num_decoder_layers"]
        d_model = self._config.transformer["d_model"]
        nhead = self._config.transformer["nhead"]
        vocab_size = self._config.vocab_size
        if self._config.transformer["dim_feedforward"]:
            dim_feedforward = self._config.transformer["dim_feedforward"]
        else:
            dim_feedforward = 1024
        if self._config.transformer["dropout"]:
            dropout = self._config.transformer["dropout"]
        else:
            dropout = 0.1
        model = Img2SeqTransformer(feature_size, extractor_name, max_seq_len, tr_extractor, num_encoder_layers,
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

    def prepare_data(self, batch_size, data_set):
        gradient_accumulate_num = self._config.gradient_accumulate_num
        nbatches = (len(data_set) + batch_size - 1) // (batch_size * gradient_accumulate_num)
        progress_bar = ProgressBar(nbatches)
        device = self._device
        train_loader = get_dataLoader(data_set, batch_size=batch_size, mode='Transformer')
        return train_loader

    def _run_train_epoch(self, train_set, val_set, lr_schedule):
        """Performs an epoch of training
                Args:
                    train_set: Dataset instance
                    val_set: Dataset instance
                    lr_schedule: LRSchedule instance that takes care of learning proc
                Returns:
                    score: (float) model will select weights that achieve the highest score
                """
        # logging
        self.model.train()
        batch_size = self._config.batch_size
        device = self._device
        accumulate_num = self._config.gradient_accumulate_num
        train_loader = self.prepare_data(batch_size, train_set)
        batch_num = len(train_loader)
        nbatches = ceil(batch_num / accumulate_num)
        progress_bar = ProgressBar(nbatches)
        losses = 0
        for i, (img, seq) in enumerate(train_loader):
            img = img.to(device)
            seq = seq.to(device)
            seq_input = seq[:, :-1]
            logits = self.model(img, seq_input)  # (batch_size, lenth, vocab_size)
            seq_out = seq[:, 1:]
            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), seq_out.reshape(-1))
            losses += loss.item()
            loss.backward()
            if ((i + 1) % accumulate_num == 0) or (i + 1 == batch_num):
                self.optimizer.step()
                self.optimizer.zero_grad()
                # update learning rate
                lr_schedule.step()
                progress_bar.update((i + 1) // accumulate_num,
                                    [("loss", loss.item()), ("lr", self.optimizer.param_groups[0]['lr'])])
        self.logger.info("- Training loss: {}".format(losses / batch_num))
        self.logger.info("- Training: {}".format(progress_bar.info))
        self.logger.info("- Config: (before evaluate, we need to see config)")
        self._config.show(fun=self.logger.info)

        # evaluation
        scores = self.evaluate(val_set)
        score = scores["Evaluate Loss"]

        return score

    def _run_evaluate_epoch(self, test_set):
        self.model.eval()
        losses = 0
        with torch.no_grad():
            batch_size = self._config.batch_size
            device = self._device
            test_loader = self.prepare_data(batch_size, test_set)
            nbatches = len(test_loader)
            progress_bar = ProgressBar(nbatches)
            for i, (img, seq) in enumerate(test_loader):
                img = img.to(device)
                seq = seq.to(device)
                seq_input = seq[:, :-1]
                logits = self.model(img, seq_input)
                seq_out = seq[:, 1:]
                loss = self.criterion(logits.reshape(-1, logits.shape[-1]), seq_out.reshape(-1))
                losses += loss.item()
                progress_bar.update(i + 1, [("loss", losses / len(test_loader))])

        self.logger.info("- Evaluating: {}".format(progress_bar.info))

        return {"Evaluate Loss": losses / len(test_loader)}

    def _run_scst(self, train_set, val_set):
        """Performs an epoch of Self-Critical Sequence Training
                Args:
                    config: Config instance
                    train_set: Dataset instance
                    val_set: Dataset instanc
                    lr_schedule: LRSchedule instance that takes care of learning proc
                Returns:
                    score: (float) model will select weights that achieve the highest score
                """
        # logging
        self.model.train()
        batch_size = self._config.batch_size
        device = self._device
        SCST_predict_mode = self._config.SCST_predict_mode
        scst_lr = self._config.SCST_lr
        train_loader = self.prepare_data(batch_size, train_set)
        nbatches = len(train_loader)
        progress_bar = ProgressBar(nbatches)
        losses = 0
        optimizer = self.getOptimizer(lr_method="adam", lr=scst_lr)
        for i, (img, seq) in enumerate(train_loader):
            img = img.to(device)
            seq = seq.to(device)
            seq_input = seq[:, :-1]
            optimizer.zero_grad()
            seq_out = seq[:, 1:]
            gts = []
            for i in range(batch_size):
                gts.append(self._vocab.decode(seq[i, :]))
            logits, sampled_seq = self.sample(img)

            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), seq_out[seq_out != PAD_ID])
            predict_seq = self.predict(img, mode=SCST_predict_mode)
            reward = SelfCritical.calculate_reward(sampled_seq, predict_seq, gts)
            loss *= reward["sample"] - reward["predict"]
            loss.backward()
            optimizer.step()
            losses += loss.item()
            progress_bar.update(i + 1, [("loss", loss), ("lr", scst_lr)])
        self.logger.info("- Training: {}".format(progress_bar.info))
        self.logger.info("- Config: (before evaluate, we need to see config)")
        self._config.show(fun=self.logger.info)

        # evaluation
        scores = self.evaluate(val_set)
        score = scores["Evaluate Loss"]

        return score

    def predict(self, img: Tensor, mode: str = "beam") -> 'list[Tensor]':
        img = img.to(self._device)
        model = self.model
        result = None
        with torch.no_grad():
            encodings = model.encode(img)
            if mode == "beam":
                result = self.beam_search.beam_decode(encode_memory=encodings)
                result = flatten_list(result)
            elif mode == "greedy":
                result = self.beam_search.greedy_decode(encode_memory=encodings)
        return result

    def sample(self, img):
        img = img.to(self._device)
        model = self.model
        encodings = model.encode(img)
        result = self.beam_search.sample_decode(encode_memory=encodings)
        return result
