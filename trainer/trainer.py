import torch
from typing import Iterable
import logging
import time

def generate_dummy_dataloader(n_samples: int):
    """
    Generate a dummy dataset for binary classification.
    """
    class1 = torch.randn(n_samples) + 1
    class2 = torch.randn(n_samples) - 1

    x = torch.cat([class1, class2]).view(-1, 1)
    y = torch.cat([torch.ones(n_samples), torch.zeros(n_samples)]).view(-1, 1)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, pin_memory=True)
    return loader
    

class Model(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, hidden_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x

# how will model kwargs get used with ray here?
def train(
    n_train_samples: int,
    n_val_samples: int,
    hidden_size: int,
    n_epochs: int,
    lr: float,
    device: str = "cpu",
):
    train_loader = generate_dummy_dataloader(n_train_samples)
    val_loader = generate_dummy_dataloader(n_val_samples)
    
    model = Model(hidden_size)
    model.to(device)

    criterion = torch.nn.functional.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    print(n_epochs)
    for epoch in range(n_epochs):
        # perform training
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # perform  validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        logging.info("Train loss: {}, Val loss: {}".format(train_loss, val_loss))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return min(val_losses)

    
