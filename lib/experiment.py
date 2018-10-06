import loss
import model
import torch.optim as optim

class Experiment():

    def __init__(self, config):
 
        model = getattr(model, config["model"])
        criterion = getattr(loss, config["criterion"])
        optimizer = getattr(optim, config["optimizer"])

        self.config = config
        self.loop = TrainingLoop(
            model = model(**config["model_args"]),
            criterion = criterion(**config["criterion_args"]),
            optimizer = optimizer(**config["optimizer_args"]),
            train_phases = config["train_phases"]
        )


    def run(self):

        self.loop.epoch(config["epochs"], self.dataloaders)
