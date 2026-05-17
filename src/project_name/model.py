"""Defines the model architecture for the project."""

import lightning as L
import torch
from torch import nn
import logging

logger = logging.getLogger(__name__)


class Model(L.LightningModule):
    """Simple fully-connected classifier for MNIST."""

    def __init__(
        self, input_size: int = 784, num_classes: int = 10, lr: float = 1e-3
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.lr = lr
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = x.view(x.size(0), -1)
        return self.network(x)

    def training_step(self, batch, batch_idx):
        """Compute loss for a single training batch and log it."""
        imgs, labels = batch
        loss = self.criterion(self(imgs), labels)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        """Return Adam optimizer with configured learning rate."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)
