import lib.dataset
import lib.loss
import lib.model
from lib.training import TrainingLoop
import torch.optim as optim

class Experiment():

    def __init__(self, config):

        self.config = config

        self.dataset = self.get_dataset(config)
        vocab_size = len(self.dataset.cols["SENT"].vocab)

        model = self.get_model(vocab_size)
        criterion = self.get_criterion()
        optimizer = self.get_optimizer(model.parameters())

        self.loop = TrainingLoop(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_phases=config["dataset_args"]["train_phases"]
        )


    def get_dataset(self, config):

        dataset = getattr(lib.dataset, self.config["dataset"])
        return dataset(**self.config["dataset_args"])


    def get_model(self, vocab_size):

        model = getattr(lib.model, self.config["model"])
        return model(vocab_size, **self.config["model_args"])


    def get_criterion(self):

        criterion = getattr(lib.loss, self.config["criterion"])
        return criterion(**self.config["criterion_args"])


    def get_optimizer(self, params):

        optimizer = getattr(optim, self.config["optimizer"])
        return optimizer(params, **self.config["optimizer_args"])


    def run(self):

        self.loop.epoch(config["epochs"], self.dataset)
