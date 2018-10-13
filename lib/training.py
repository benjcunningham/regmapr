import torch
import numpy as np

class TrainingLoop():
    """Training loop

    Args:
        model: nn.Module object
        criterion: Loss function
        optimizer: Optimizer object
        train_phases: Phases where training mode should be used
    """

    def __init__(self, model, criterion, optimizer, train_phases):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_phases = train_phases


    def epoch(self, epochs, dataloaders):
        """Train multiple epochs

        Args:
            epochs: Number of epochs
            dataloaders: Dictionary of DataLoader objects
        """

        for epoch in range(epochs):
            for phase in dataloaders.keys():

                self.phase(epoch, phase, dataloaders[phase])


    def phase(self, epoch, phase, dataloader):
        """Train one phase of an epoch

        Args:
            phase: Name of phase
            dataloader: DataLoader object
        """

        self.model.train(phase in self.train_phases)

        losses = []
        for i, data in enumerate(dataloader):

            self.optimizer.zero_grad()

            target = data.target.unsqueeze(1)
            pred = self.model(data.s1, data.s2)
            loss = self.criterion(pred, target)

            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            batch = [i, len(dataloader)]
            status(phase, epoch, batch, np.mean(losses))


def status(phase, epoch, batch, loss):
    """Print training status

    Args:
        phase: Name of phase
        epoch: Current epoch
        batch: List with index of current minibatch and batch size
        loss: Current loss
    """

    print(f"[{phase}] Epoch: {epoch} " +
          f"({batch[0] + 1}/{batch[1]}) Loss: {loss:.4f}",
          end="\n" if (batch[0] + 1) == batch[1] else "\r")

