import torch

from base import BaseModel
from data_gen import get_dataLoader
from model.Img2Seq import Img2Seq
from pkg.utils.ProgBar import ProgressBar
from pkg.utils.BeamSearch import beam_decode
from pkg.utils.BeamSearch import greedy_decode
from pkg.utils.utils import num_param

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2


class Img2InchiModel(BaseModel):
    def __init__(self, config, output_dir, vocab):
        super(Img2InchiModel, self).__init__(config, output_dir)
        self._vocab = vocab
        self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
            progress_bar.update(i + 1, [("loss", loss), ("lr", lr_schedule.lr)])

            # update learning rate
            lr_schedule.update(batch_no=epoch * nbatches + i)

        self.logger.info("- Training: {}".format(progress_bar.info))
        self.logger.info("- Config: (before evaluate, we need to see config)")
        config.show(fun=self.logger.info)

        # evaluation
        scores = self.evaluate(val_set)
        score = scores["perplexity"]
        lr_schedule.update(score=score)

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
        encodings = model.encode(img)
        result = None
        if mode == "beam":
            result = beam_decode(decoder=self.model.decoder, encodings=encodings, beam_width=10, topk=1,
                                 max_len=max_len)
        elif mode == "greedy":
            seq = torch.ones(1, 1).fill_(SOS_ID).type(torch.long).to(self._device)
            result = greedy_decode(self.model.decoder, encodings, seq)
        if result.ndim == 3:
            decoded_result = []
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    decoded_result.append(self._vocab.decode(result[i, j, :]))
            decoded_tensor = torch.Tensor(decoded_result)
            decoded_tensor.view(result.shape[0], result.sjape[1])
            return decoded_tensor
        elif result.ndim == 2:
            decoded_result = []
            for i in range(result.shape[0]):
                decoded_result.append(self._vocab.decode(result[i, :]))
            decoded_tensor = torch.Tensor(decoded_result)
            return decoded_tensor

    def sample(self, encodings):
        # TODO implement sampling method
        pass
