"""Module for training the model."""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from project_name.model import Model


def train(lr: float = 1e-3, epochs: int = 5, batch_size: int = 64) -> None:
    """Train the model on MNIST and save checkpoint."""
    transform = transforms.ToTensor()
    dataset = datasets.MNIST("data/raw", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, labels in dataloader:
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "models/model.pt")
    print("Model saved to models/model.pt")


if __name__ == "__main__":
    train()
