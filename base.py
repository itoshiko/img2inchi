import logging
import time

from tensorboardX import SummaryWriter

import torch

from pkg.utils.general import get_logger, init_dir
from visualization.structure import generate_structure


class BaseModel(object):
    def __init__(self, config, output_dir, need_output=True):
        self._config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.multi_gpu = config.multi_gpu and torch.cuda.device_count() > 1
        if need_output:
            self._output_dir = output_dir
            self._init_relative_path(output_dir)
            self.logger = get_logger(log_path=self._log_dir)
        else:
            self.logger = logging.getLogger()

    def _init_relative_path(self, output_dir):
        init_dir(output_dir)
        self._model_dir = output_dir + "/" + self._config.instance
        init_dir(self._model_dir)
        self._log_dir = self._model_dir + "/logs"
        init_dir(self._log_dir)
        self._config_export_path = self._model_dir
        self.write_path = self._model_dir + "/runs/" + time.strftime("%Y-%m-%d %H.%M.%S", time.localtime())

    def build_train(self, config=None):
        self.logger.info("- Building model...")
        self._init_model(config.model_name, self.device)
        self._init_optimizer(config.lr_method, config.lr_init)
        self._init_scheduler(config.lr_scheduler)
        self._init_criterion(config.criterion_method)
        if self.multi_gpu:
            self._init_multi_gpu()
        self._init_writer()
        self.logger.info("- Config: ")
        self._config.show(fun=self.logger.info)

        self.logger.info("- done.")

    def _init_model(self, model_name="transformer"):
        self.logger.info("   - " + model_name)
        self.logger.info("   - " + str(self.device))
        self.model = self.getModel()
        self.model = self.model.to(self.device)

        # if model exists, load it
        if self.is_resume:
            self.restore(map_location=str(self.device))


    def _init_optimizer(self, lr_method="adam", lr=1e-3):
        """Defines self.optimizer that performs an update on a batch
        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: init learning rate (initial value)
        """
        # 1. optimizer
        _lr_m = lr_method.lower()  # lower to make sure
        print("  - optimizer: " + _lr_m)
        self.optimizer = self.getOptimizer(_lr_m, lr)
        if self.is_resume:
            self.optimizer.load_state_dict(self.old_model["optimizer"])

    def _init_scheduler(self, lr_scheduler="CosineAnnealingLR"):
        """Defines self.scheduler that performs an update on a batch
        Args:
            lr_scheduler: (string) learning rate schedule method, for example "CosineAnnealingLR"
        """
        # 2. scheduler
        print("  - lr_scheduler: " + lr_scheduler)
        self.scheduler = self.getLearningRateScheduler(lr_scheduler)

    def _init_criterion(self, criterion_method="CrossEntropyLoss"):
        """Defines self.criterion that performs an update on a batch
        Args:
            criterion_method: (string) criterion method, for example "CrossEntropyLoss"
        """
        # 3. criterion
        print("  - criterion: " + criterion_method)
        self.criterion = self.getCriterion(criterion_method)

    def _init_multi_gpu(self):
        device_ids = range(torch.cuda.device_count())
        self.model = torch.nn.DataParallel(self.model, device_ids = device_ids)
        self.logger.info("  - multi-gpu: cuda:", *device_ids)

    def _init_writer(self):
        self.writer = SummaryWriter(log_dir=self.write_path)

    # ! MUST OVERWRITE
    def getModel(self):
        """return your Model
        Args:
        Returns:
            your model that inherits from torch.nn
        """
        raise NotImplementedError("return your model ({}) that inherits from torch.nn")

    def getOptimizer(self, lr_method="adam", lr=1e-3):
        if lr_method == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        elif lr_method == 'adamax':
            return torch.optim.Adamax(self.model.parameters(), lr=lr)
        elif lr_method == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError("Unknown Optimizer {}".format(lr_method))

    def getLearningRateScheduler(self, lr_scheduler="CosineAnnealingLR"):
        if lr_scheduler == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=4e-08)
        else:
            raise NotImplementedError("Unknown Learning Rate Scheduler {}".format(lr_scheduler))

    def getCriterion(self, criterion_method="CrossEntropyLoss"):
        if criterion_method == 'CrossEntropyLoss':
            return torch.nn.CrossEntropyLoss()
        elif criterion_method == 'MSELoss':
            return torch.nn.MSELoss()
        elif criterion_method == 'BCEWithLogitsLoss':
            return torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError("Unknown Criterion Method {}".format(criterion_method))

    # 3. save and restore
    def restore(self, map_location='cpu'):
        """Reload weights into session
        Args:
            map_location: 'cpu' or 'gpu:0'
        """
        self.logger.info("- Reloading the latest trained model...")
        self.old_model = torch.load(self._model_path, map_location)
        self.model.load_state_dict(self.old_model["net"])

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
        torch.save(checking_point, self._model_path)
        self._config.save(self._config_export_path)
        self.logger.info("- Saved model in {}".format(self._model_dir))

    def write_loss(self, step: int, loss: float):
        self.writer.add_scalar(tag="loss", scalar_value=loss, global_step=step)

    def write_eval(self, result: dict):
        for key in result:
            self.writer.add_scalar(tag='eval_' + str(key), scalar_value=result[key], global_step=self.now_epoch)

    # 4. train and evaluate
    def train(self, train_set, val_set):
        """Global training procedure
        Calls method self.run_epoch and saves weights if score improves.
        All the epoch-logic including the lr_schedule update must be done in
        self.run_epoch
        Args:
            config: Config instance contains params as attributes
            train_set: Dataset instance
            val_set: Dataset instance
        Returns:
            best_score: (float)
        """
        config = self._config
        best_score = None
        # generate_structure(self.model, config)
        if self.is_resume:
            self.now_epoch = self.old_model["epoch"]
        else:
            self.now_epoch = 0
        # all restored, delete attr to release memory
        if hasattr(self, "old_model"):
            delattr(self, "old_model")
        
        model = self.model
        optimizer = self.optimizer
        for epoch in range(self.now_epoch, config.n_epochs):
            # logging
            tic = time.time()
            self.logger.info("Epoch {:}/{:}".format(epoch + 1, config.n_epochs))

            # epoch
            score = self._run_train_epoch(model, optimizer, train_set, val_set, self.scheduler)
            self.now_epoch += 1

            # save weights if we have new best score on eval
            if best_score is None or score >= best_score:  # abs(score-0.5) <= abs(best_score-0.5):
                best_score = score
                self.logger.info("- New best score ({:04.2f})!".format(best_score))
                self.save()

            # logging
            toc = time.time()
            self.logger.info("- Elapsed time: {:04.2f}, learning rate: {:04.5f}"
                             .format(toc - tic, self.optimizer.param_groups[0]['lr']))

        return best_score

    def evaluate(self, test_set):
        """Evaluates model on test set
        Calls method run_evaluate on test_set and takes care of logging
        Args:
            test_set: instance of class Dataset
        Return:
            scores: (dict) scores["acc"] = 0.85 for instance
        """
        self.logger.info("- Evaluating...")
        evaluate_loss = self._run_evaluate_epoch(test_set)  # evaluate
        msg = " || ".join([" {} is {:04.2f} ".format(k, v) for k, v in evaluate_loss.items()])
        self.logger.info("- Eval: {}".format(msg))

        return evaluate_loss

    def _auto_backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ! MUST OVERWRITE
    def _run_train_epoch(self, model, optimizer, train_set, val_set, lr_schedule):
        """Model_specific method to overwrite
        Performs an epoch of training
        Args:
            train_set: Dataset instance
            val_set: Dataset instance
            lr_schedule: LRSchedule instance that takes care of learning proc
        Returns:
            score: (float) model will select weights that achieve the highest score
        Alert:
            you can use the method below to simplify your code
            _auto_backward(self, loss)
        """
        raise NotImplementedError("Performs an epoch of training")

    # ! MUST OVERWRITE
    def _run_evaluate_epoch(self, test_set):
        """Model-specific method to overwrite
        Performs an epoch of evaluation
        Args:
            test_set: Dataset instance
        Returns:
            scores: (dict) scores["acc"] = 0.85 for instance
        """
        raise NotImplementedError("Performs an epoch of evaluation")

    # ! MUST OVERWRITE
    def _run_scst(self, train_set, val_set):
        """Model-specific method to overwrite
        Performs an epoch of Self-Critical Sequence Training
        Args:
            train_set:
            val_set:
        Returns:
            scores: (dict) scores["acc"] = 0.85 for instance
        """
        raise NotImplementedError("Performs an epoch of evaluation")

    # ! MUST OVERWRITE
    def predict(self, img, max_len=200, mode="beam"):
        """Model-specific method to overwrite
                Performs an epoch of evaluation
                Args:
                    img: images to predict
                    max_len: max length for inchi string
                    mode: search mode, beam or greedy
                Returns:
                    scores: (dict) scores["acc"] = 0.85 for instance
                """
        raise NotImplementedError("Performs prediction")

    # ! MUST OVERWRITE
    def sample(self, encodings):
        """Model-specific method to overwrite
                Performs sampling when decoding
                Args:
                    encodings: output from network encoder
                Returns:
                    ws: sampling result
                """
        raise NotImplementedError("Performs sampling")
