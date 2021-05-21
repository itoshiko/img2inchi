import torch

import model.SelfCritical as SelfCritical

from base import BaseModel
from data_gen import get_dataLoader
from model.Transformer import Img2SeqTransformer
from pkg.utils.ProgBar import ProgressBar
from pkg.utils.BeamSearchTransformer import BeamSearchTransformer
from pkg.utils.BeamSearch import greedy_decode
from pkg.utils.utils import num_param


PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


class Img2InchiTransformerModel(BaseModel):
    def __init__(self, config, output_dir, vocab):
        super(Img2InchiTransformerModel, self).__init__(config, output_dir)
        self._vocab = vocab
        self._device = config.device
        if self.device is None:
            self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.max_len = config.max_seq_len

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
                                 (min(step ** (-0.5), step * (warmup_steps ** (-0.5))))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_sc)
        else:
            return super().getLearningRateScheduler(lr_scheduler)

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
        progress_bar = ProgressBar(nbatches)
        self.model.train()
        losses = 0
        device = self._device
        train_loader = get_dataLoader(train_set, batch_size=batch_size, mode='Transformer')
        for i, (img, seq) in enumerate(train_loader):
            img = img.to(device)
            seq = seq.to(device)
            seq_input = seq[:, :-1]
            logits = self.model(img, seq_input)  # (batch_size, lenth, vocab_size)
            self.optimizer.zero_grad()
            seq_out = seq[:, 1:]
            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), seq_out.reshape(-1))
            loss.backward()
            self.optimizer.step()
            losses += loss.item()
            progress_bar.update(i + 1, [("loss", loss), ("lr", lr_schedule.lr)])
            # update learning rate
            lr_schedule.update(batch_no=epoch * nbatches + i)
        self.logger.info("- Training: {}".format(progress_bar.info))
        self.logger.info("- Config: (before evaluate, we need to see config)")
        config.show(fun=self.logger.info)

        # evaluation
        scores = self.evaluate(val_set)
        score = scores["Evaluate Loss"]
        lr_schedule.update(score=score)

        return score

    def _run_evaluate_epoch(self, test_set):
        self.model.eval()
        losses = 0
        with torch.no_grad():
            batch_size = self._config.batch_size
            test_loader = get_dataLoader(test_set, batch_size=batch_size, mode='Transformer')
            nbatches = len(test_loader)
            prog = ProgressBar(nbatches)

            for i, (img, seq) in enumerate(test_loader):
                img = img.to(self._device)
                seq = seq.to(self._device)
                seq_input = seq[:, :-1]
                logits = self.model(img, seq_input)
                seq_out = seq[:, 1:]
                loss = self.criterion(logits.reshape(-1, logits.shape[-1]), seq_out.reshape(-1))
                losses += loss.item()
                prog.update(i + 1, [("loss", losses / len(test_loader))])

        self.logger.info("- Evaluating: {}".format(prog.info))

        return {"Evaluate Loss": losses / len(test_loader)}

    def _run_scst_epoch(self, config, train_set, val_set, epoch, lr_schedule):
        """Performs an epoch of Self-Critical Sequence Training
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
        SCST_predict_mode = config.SCST_predict_mode
        nbatches = (len(train_set) + batch_size - 1) // batch_size
        progress_bar = ProgressBar(nbatches)
        self.model.train()
        losses = 0
        device = self._device
        train_loader = get_dataLoader(train_set, batch_size=batch_size, mode='Transformer')
        for i, (img, seq) in enumerate(train_loader):
            img = img.to(device)
            seq = seq.to(device)
            seq_input = seq[:, :-1]
            logits = self.model(img, seq_input)  # (batch_size, lenth, vocab_size)
            self.optimizer.zero_grad()
            seq_out = seq[:, 1:]
            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), seq_out.reshape(-1))
            gts = []
            for i in range(batch_size):
                gts.append(self._vocab.decode(seq[i, :]))
            sampled_seq = self.sample(img)
            predict_seq = self.predict(img, mode=SCST_predict_mode)
            reward = SelfCritical.calculate_reward(sampled_seq, predict_seq, gts)
            r = reward["sample"] - reward["predict"]
            loss = SelfCritical.SelfCritical.apply(loss, r)
            loss.backward()
            self.optimizer.step()
            losses += loss.item()
            progress_bar.update(i + 1, [("loss", loss), ("lr", lr_schedule.lr)])
            # update learning rate
            lr_schedule.update(batch_no=epoch * nbatches + i)
        self.logger.info("- Training: {}".format(progress_bar.info))
        self.logger.info("- Config: (before evaluate, we need to see config)")
        config.show(fun=self.logger.info)

        # evaluation
        scores = self.evaluate(val_set)
        score = scores["Evaluate Loss"]
        lr_schedule.update(score=score)

        return score

    def predict(self, img, mode="beam"):
        img = img.to(self._device)
        model = self.model
        encodings = model.encode(img)
        result = None
        if mode == "beam":
            beam_search = BeamSearchTransformer(decoder=self.model.decoder, device=self._device, beam_width=10,
                                            topk=1, max_len=self.max_len, max_batch=100)
            result = beam_search.beam_decode(encode_memory=encodings)
        elif mode == "greedy":
            seq = torch.ones(1, 1).fill_(SOS_ID).type(torch.long).to(self._device)
            result = greedy_decode(self.model.decoder, encodings, seq, False)
        if result.ndim == 3:
            decoded_result = []
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    decoded_result.append(self._vocab.decode(result[i, j, :]))
            decoded_tensor = torch.Tensor(decoded_result)
            decoded_tensor.view(result.shape[0], result.shape[1])
            return decoded_tensor
        elif result.ndim == 2:
            decoded_result = []
            for i in range(result.shape[0]):
                decoded_result.append(self._vocab.decode(result[i, :]))
            decoded_tensor = torch.Tensor(decoded_result)
            return decoded_tensor

    def sample(self, img):
        img = img.to(self._device)
        model = self.model
        encodings = model.encode(img)
        seq = torch.ones(1, 1).fill_(SOS_ID).type(torch.long).to(self._device)
        result = greedy_decode(self.model.decoder, encodings, seq, True)
        sampled_result = []
        if result.ndim == 3:
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    sampled_result.append(self._vocab.decode(result[i, j, :]))
            sampled_tensor = torch.Tensor(sampled_result)
            sampled_tensor.view(result.shape[0], result.shape[1])
            return sampled_tensor
        elif result.ndim == 2:
            for i in range(result.shape[0]):
                sampled_result.append(self._vocab.decode(result[i, :]))
            sampled_tensor = torch.Tensor(sampled_result)
            return sampled_tensor

