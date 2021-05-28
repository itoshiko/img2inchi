
from typing import Iterable, Union
from base import BaseModel
from data_gen import get_dataLoader
from pkg.utils.BeamSearchLSTM import BeamSearchLSTM
from pkg.utils.BeamSearchTransformer import BeamSearchTransformer
from math import ceil

import torch
import Levenshtein
from torch import Tensor

from pkg.utils.ProgBar import ProgressBar
from pkg.utils.utils import flatten_list, split_list


class Img2InchiModel(BaseModel):
    def __init__(self, config, output_dir, vocab, need_output=True):
        super(Img2InchiModel, self).__init__(config, output_dir, need_output=need_output)
        self.model_name = config.model_name
        self.max_len = config.max_seq_len
        self._vocab = vocab
        self.vocab_size = vocab.size

    def build_train(self, config=None):
        self.logger.info("- Building model...")
        self._init_model(config.model_name)
        self._init_optimizer(config.lr_method, config.lr_init)
        self._init_scheduler(config.lr_scheduler)
        self._init_criterion(config.criterion_method)
        self._init_beamSearch(config)
        if self.multi_gpu:
            self._init_multi_gpu()
        self._init_writer()
        self.logger.info("- Config: ")
        self._config.show(fun=self.logger.info)

        self.logger.info("- done.")

    def build_pred(self, model_path, config=None):
        self.logger.info("- Building model...")
        self.logger.info("   - " + config.model_name)
        self.logger.info("   - " + str(self.device))
        self.model = self.getModel()
        self.model = self.model.to(self.device)
        model_from_disk = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_from_disk["net"])
        self._init_beamSearch(config)
        self.logger.info("- done.")

    def _init_beamSearch(self, config):
        self.beam_search = self.getBeamSearch(model_name=config.model_name, device=self.device, 
                                            beam_width=config.beam_width, max_len=self.max_len, max_batch=config.batch_size)

    def getBeamSearch(self, model_name='transformer', device='cpu', beam_width=10, max_len=10, max_batch=100):
        if model_name == 'transformer':
            beam_search = BeamSearchTransformer(tf_model=self.model, device=device, beam_width=beam_width,
                                                 topk=1, max_len=max_len, max_batch=max_batch)
        elif model_name == 'lstm':
            beam_search = BeamSearchLSTM(decoder=self.model.decoder, device=device, beam_width=beam_width,
                                                 topk=1, max_len=max_len, max_batch=max_batch)
        else:
            raise NotImplementedError("Unknown model name")
        return beam_search

    def prepare_data(self, batch_size, data_set, shuffle=True):
        train_loader = get_dataLoader(data_set, batch_size=batch_size, shuffle=shuffle, 
                                        num_workers=self._config.dataloader_num_workers, model_name=self.model_name)
        return train_loader
        
    def scst(self, train_set, val_set):
        self._run_scst(train_set, val_set)

    def _run_train_epoch(self, model, optimizer, train_set, val_set, lr_schedule):
        """Performs an epoch of training
                Args:
                    train_set: Dataset instance
                    val_set: Dataset instance
                    lr_schedule: LRSchedule instance that takes care of learning proc
                Returns:
                    score: (float) model will select weights that achieve the highest score
                """
        # logging
        losses = 0
        model.train()
        batch_size = self._config.batch_size
        device = self.device
        accumulate_num = self._config.gradient_accumulate_num
        train_loader = self.prepare_data(batch_size, train_set)
        batch_num = len(train_loader)
        nbatches = ceil(batch_num / accumulate_num)
        progress_bar = ProgressBar(nbatches)
        optimizer.zero_grad()
        for i, (img, seq) in enumerate(train_loader):
            img = img.to(device)
            seq = seq.to(device)
            seq_input = seq[:, :-1]
            logits = model(img, seq_input)  # (batch_size, lenth, vocab_size)
            seq_out = seq[:, 1:]
            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), seq_out.reshape(-1))
            loss.backward()
            losses += loss.item()
            if ((i + 1) % accumulate_num == 0) or (i + 1 == batch_num):
                optimizer.step()
                optimizer.zero_grad()
                # update learning rate
                lr_schedule.step()
                progress_bar.update(ceil((i + 1) / accumulate_num), [("loss", loss.item()), 
                                    ("lr", optimizer.param_groups[0]['lr'])])
                if (i + 1) % (50 * accumulate_num) == 0:
                    score = torch.mean(self.calculate_reward(torch.max(logits, dim=-1)[1], seq)).item()
                    self.write_train(self.now_epoch * nbatches + ceil((i + 1) / accumulate_num), 
                                    {"loss": loss.item(), "score": score})
        self.logger.info("- Training: {}".format(progress_bar.info))

        # evaluation
        scores = self.evaluate(val_set)
        self.write_eval(scores)
        score = scores["Score"]

        return score

    def _run_evaluate_epoch(self, test_set):
        self.model.eval()
        losses = 0
        scores = 0
        batch_size = self._config.eval_batch_size
        device = self.device
        test_loader = self.prepare_data(batch_size, test_set, False)
        nbatches = len(test_loader)
        progress_bar = ProgressBar(nbatches)
        with torch.no_grad():
            for i, (img, seq) in enumerate(test_loader):
                img = img.to(device)
                seq = seq.to(device)
                seq_input = seq[:, :-1]
                logits = self.model(img, seq_input)
                seq_out = seq[:, 1:]
                loss = self.criterion(logits.reshape(-1, logits.shape[-1]), seq_out.reshape(-1))
                losses += loss.item()
                
                score = torch.mean(self.calculate_reward(torch.max(logits, dim=-1)[1], seq_out)).item()
                scores += score
                progress_bar.update(i + 1, [("loss", loss.item()), ("score", score)])

            predict_scores = 0
            num_predicted = 0
            for i, (img, seq) in enumerate(test_loader):
                predict_score = torch.sum(self.calculate_reward(self.predict(img=img, mode='greedy'), 
                                        self._vocab.decode(seq))).item()
                predict_scores += predict_score
                num_predicted += img.shape[0]
                if num_predicted >= 1000:
                    break

        self.logger.info("- Evaluating: {}".format(progress_bar.info))

        return {"Loss": losses / len(test_loader), "Score": scores / len(test_loader), 
                "Predict_score": predict_scores / num_predicted}

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
        self.model.scst()
        batch_size = self._config.batch_size
        device = self.device
        scst_lr = self._config.SCST_lr
        train_loader = self.prepare_data(batch_size, train_set)
        nbatches = len(train_loader)
        progress_bar = ProgressBar(nbatches)
        losses = 0
        optimizer = self.getOptimizer(lr_method="adam", lr=scst_lr)
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        for t in range(50, 0, -5):
            for i, (img, seq) in enumerate(train_loader):
                img = img.to(device)
                seq = seq.to(device)
                optimizer.zero_grad()
                seq_out = seq[:, 1:]
                logits, sampled_seq = self.sample(img=img, gts=seq_out, forcing_num=0)
                loss = criterion(logits.transpose(1, 2).contiguous(), seq_out)
                gts = self._vocab.decode(seq)
                more_sample = self.sample_decode(img, num_samples=10)
                sampled_reward = self.calculate_reward(sampled_seq, gts)
                baseline = torch.stack([self.calculate_reward(sample, gts) for sample in more_sample])
                baseline = torch.mean(baseline, dim=0)
                loss = torch.mean(torch.mean(loss * (sampled_reward - baseline).unsqueeze(-1), dim=1))
                loss.backward()
                optimizer.step()
                losses += loss.item()
                self.model.clear_cache()
                pr = self.predict(img, 'greedy')
                progress_bar.update(i + 1, [("loss", loss.item()), ("lr", optimizer.param_groups[0]['lr']), 
                                    ("reward", torch.mean(self.calculate_reward(pr, gts)).item())])
            self.logger.info("- Training: {}".format(progress_bar.info))
            self.logger.info("- Config: (before evaluate, we need to see config)")
            self._config.show(fun=self.logger.info)

            # evaluation
            scores = self.evaluate(val_set)
            score = scores["Evaluate Loss"]

        return score

    def predict(self, img: Tensor, mode: str = "greedy") -> 'Tensor':
        img = img.to(self.device)
        if self.multi_gpu:
            model = self.model.module
        else:
            model = self.model
        result = None
        with torch.no_grad():
            encodings = model.encode(img)
            if mode == "beam":
                result = self.beam_search.beam_decode(encode_memory=encodings)
                result = flatten_list(result)
            elif mode == "greedy":
                result = self.beam_search.greedy_decode(encode_memory=encodings)
        return torch.stack(result)

    def sample(self, img: Tensor, gts: Tensor, forcing_num: int):
        img = img.to(self.device)
        model = self.model
        with torch.no_grad():
            encodings = model.encode(img)
        result = self.beam_search.sample(encode_memory=encodings, gts=gts, forcing_num=forcing_num, vocab_size=self.vocab_size)
        return result
    
    def sample_decode(self, img: Tensor, num_samples: int=5):
        img = img.to(self.device)
        model = self.model
        with torch.no_grad():
            enc = model.encode(img)
            batch_size = enc.shape[0]
            encodings = torch.zeros((batch_size * num_samples, enc.shape[1], enc.shape[2]), device=self.device)
            for k in range(batch_size):
                encodings[k::batch_size] = enc[k]
            del enc
            result = self.beam_search.sample_decode(encode_memory=encodings)
        return split_list(result, batch_size)

    def calculate_reward(self, seq: Tensor, gt: Iterable[str]):
        """
                Calculate Levenshtein distance of sample sequence and predict sequence
                Args:
                    seq: sequence from sample
                    gt: ground-truth sequence
                Returns:
                    reward: (dict) reward["sample"], reward["predict"]
                """
        if isinstance(gt, Tensor):
            gt = self._vocab.decode(gt)
        with torch.no_grad():
            batch_size = len(gt)
            seq = self._vocab.decode(seq)
            sample_reward = torch.zeros(batch_size, device=self.device)
            for i in range(batch_size):
                sample_reward[i] = 1 - (Levenshtein.distance(seq[i], gt[i]) / len(gt[i]))
        return sample_reward