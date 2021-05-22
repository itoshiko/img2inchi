import torch

from base import BaseModel
from data_gen import get_dataLoader
from model.Img2Seq import Img2Seq
from pkg.utils.ProgBar import ProgressBar
from pkg.utils.BeamSearchLSTM import BeamSearchLSTM
from pkg.utils.utils import flatten_list, num_param

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


class Img2InchiModel(BaseModel):
    def __init__(self, config, output_dir, vocab):
        super(Img2InchiModel, self).__init__(config, output_dir)
        self._vocab = vocab
        self._device = config.device
        if self._device is None:
            self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.max_len = config.max_seq_len
        self.beam_search = BeamSearchLSTM(decoder=self.model.decoder, device=self._device, beam_width=config.beam_width,
                                                topk=1, max_len=self.max_len, max_batch=config.batch_size)

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
        loss_mode = config.loss_mode
        nbatches = (len(train_set) + batch_size - 1) // batch_size
        progress_bar = ProgressBar(nbatches)
        self.model.train()
        losses = 0
        train_loader = get_dataLoader(train_set, batch_size=batch_size, mode='Img2Seq')

        for i, (img, seq) in enumerate(train_loader):
            # img = torch.FloatTensor(img)
            # seq = torch.LongTensor(seq)  # (N,)
            img = img.to(self._device)
            seq = seq.to(self._device)
            seq_input = seq[:, :-1]
            logits = self.model(img, seq_input)
            self.optimizer.zero_grad()
            seq_out = seq[:, 1:]
            if loss_mode == "SCST":
                # TODO implement SCST algorithm
                pass
            else:
                loss = self.criterion(logits.reshape(-1, logits.shape[-1]), seq_out.reshape(-1))
            loss.backward()
            self.optimizer.step()
            losses += loss.item()
            progress_bar.update(i + 1, [("loss", loss), ("lr", self.optimizer.param_groups[0]['lr'])])

            # update learning rate
            lr_schedule.step()

        self.logger.info("- Training: {}".format(progress_bar.info))
        self.logger.info("- Config: (before evaluate, we need to see config)")
        config.show(fun=self.logger.info)

        # evaluation
        scores = self.evaluate(val_set)
        score = scores["Evaluate Loss"]

        return score

    def _run_evaluate_epoch(self, test_set):
        self.model.eval()
        losses = 0
        with torch.no_grad():
            nbatches = len(test_set)
            batch_size = self._config.batch_size
            prog = ProgressBar(nbatches)
            test_loader = get_dataLoader(test_set, batch_size=batch_size, mode='Img2Seq')

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

    def predict(self, img, max_len=200, mode="beam"):
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

    def sample(self, encodings):
        # TODO implement sampling method
        pass
