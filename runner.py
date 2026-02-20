import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from models.rvenet.all_models import model_union
from dicom_loader import DicomLoader

def compute_loss(
    model: model_union,
    model_output: torch.Tensor,
    target_values: torch.Tensor,
    is_normalize: bool = True,
) -> torch.Tensor:
    """
    Computes the loss between the model output and the target values for regression tasks.

    Args:
    -   model: a model (which inherits from nn.Module) with a defined loss criterion
    -   model_output: the raw predictions output by the net
    -   target_values: the ground truth values for regression
    -   is_normalize: bool flag indicating that loss should be divided by the batch size
    Returns:
    -   the computed loss value
    """
    # Use the model's loss criterion, e.g., nn.MSELoss or similar, to calculate the loss
    loss = model.loss_criterion(model_output, target_values)

    # If is_normalize is True and the loss is not already normalized, divide by batch size
    if is_normalize and loss.dim() > 0:
        batch_size = model_output.size(0)
        loss = loss / batch_size

    return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class Trainer:
    """Class that stores model training metadata."""

    def __init__(
        self,
        environment: dict,
        model: model_union,
        optimizer: Optimizer,
        model_dir: str,
        train_data_transforms: transforms.Compose,
        val_data_transforms: transforms.Compose,
        batch_size: int = 100,
        load_from_disk: bool = True,
        cuda: bool = False,
        num_augmented_features = 0,
    ) -> None:
        self.model_dir = model_dir

        self.model = model

        self.cuda = cuda
        if cuda:
            self.model.cuda()

        num_workers = 1
        dataloader_args = {"num_workers": num_workers, "pin_memory": True} if cuda else {"num_workers": num_workers}

        self.train_dataset = DicomLoader(
            environment, split="train", transform=train_data_transforms, deal_with_pt=True
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
        )

        self.val_dataset = DicomLoader(
            environment, split="validation", transform=val_data_transforms, deal_with_pt=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
        )

        self.num_augmented_features = num_augmented_features
        self.optimizer = optimizer

        self.train_loss_history = []
        self.validation_loss_history = []

        self.actual_values = None
        self.predicted_values = None

        # create the model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # check if checkpoint exists
        if os.path.exists(os.path.join(self.model_dir, "checkpoint.pt")) and load_from_disk:
            checkpoint = torch.load(os.path.join(self.model_dir, "checkpoint.pt"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.model.train()

    def save_model(self) -> None:
        """
        Saves the model state and optimizer state on the dict
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.model_dir, "checkpoint.pt"),
        )

    def run_training_loop(self, num_epochs: int) -> None:
        """Train for num_epochs, and validate after every epoch."""
        for epoch_idx in range(num_epochs):

            train_loss = self.train_epoch(epoch_idx)

            self.train_loss_history.append(train_loss)

            val_loss = self.validate()
            self.validation_loss_history.append(val_loss)

            print(
                f"Epoch:{epoch_idx + 1}"
                + f" Train Loss:{train_loss:.4f}"
                + f" Val Loss: {val_loss:.4f}"
            )

    def train_epoch(self, epoch_idx: int) -> float:
        """Implements the main training loop for regression."""
        self.model.train()

        train_loss_meter = AverageMeter("train loss")
        sub_epoch=0
        
        # loop over each minibatch
        for (dictionary, label) in self.train_loader:
            x=dictionary['video_tensor']
            y=label
            if self.cuda:
                x = x.cuda()
                y = y.cuda()

            n = x.shape[0]

            if self.num_augmented_features > 0:
                extra_features=dictionary['extra_features']
                if self.cuda:
                    extra_features=extra_features.cuda()

                predictions = self.model(x,extra_features)
            else:
                predictions = self.model(x)

            #TODO try R^2 for regression, MSE, RMSE etc

            # Compute regression loss
            batch_loss = compute_loss(self.model, predictions, y, is_normalize=True)

            train_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

            # Backpropagation and optimization step
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # Print for debugging
            # print(
            #     f"Epoch:{epoch_idx + 1}"+f" SubEpoch:{sub_epoch + 1}"
            #     + f" Train Loss:{train_loss_meter.val:.4f}"
            # )
            sub_epoch+=1

        return train_loss_meter.avg

    def validate(self) -> float:
        """Evaluate on held-out split (either val or test) for regression"""
        self.model.eval()

        val_loss_meter = AverageMeter("val loss")

        # Loop over the validation set
        with torch.no_grad():  # Disable gradient calculation for validation
            for (dictionary, label) in self.val_loader:
                x=dictionary['video_tensor']
                y=label
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()

                n = x.shape[0]

                if self.num_augmented_features > 0:
                    extra_features=dictionary['extra_features']
                    if self.cuda:
                        extra_features=extra_features.cuda()

                    predictions = self.model(x,extra_features)
                else:
                    predictions = self.model(x)

                # Compute regression loss
                batch_loss = compute_loss(self.model, predictions, y, is_normalize=True)
                val_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

        return val_loss_meter.avg

    def store_predictions(self) -> None:
        """Plot predicted vs actual values"""
        self.model.eval()

        self.predicted_values = []
        self.actual_values = []

        with torch.no_grad():
            for (dictionary, label) in self.val_loader:
                x=dictionary['video_tensor']
                y=label
                if self.cuda:
                    x = x.cuda()
                    y = y.cuda()

                if self.num_augmented_features > 0:
                    extra_features=dictionary['extra_features']
                    if self.cuda:
                        extra_features=extra_features.cuda()

                    predictions = self.model(x,extra_features)
                else:
                    predictions = self.model(x)

                self.predicted_values.extend(predictions.cpu().numpy())
                self.actual_values.extend(y.cpu().numpy())

    def plot_predictions(self) -> None:
        if self.predicted_values is None or self.actual_values is None:
            self.store_predictions()

        plt.figure()
        plt.scatter(self.predicted_values, self.actual_values)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        # plt.axis('scaled')
        plt.savefig(os.path.join(self.model_dir, "predictions.png"))

        plt.show()

    def plot_loss_history(self) -> None:
        """Plots the loss history"""
        plt.figure()
        epoch_idxs = range(len(self.train_loss_history))

        plt.plot(epoch_idxs, self.train_loss_history, "-b", label="training")
        plt.plot(epoch_idxs, self.validation_loss_history, "-r", label="validation")
        plt.title("Loss history")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(self.model_dir, "loss_history.png"))

        plt.show()
