import click
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from models import CVAE, VAE
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Stack(nn.Module):
    def __init__(self, channels, height, width):
        super(Stack, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)


class Cond(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hiddenc2mu = nn.Linear(10, hidden_size)

    def forward(self, y):
        y_one_hot = F.one_hot(y, num_classes=10).float()
        mu = self.hiddenc2mu(y_one_hot)
        return mu, torch.ones_like(mu)


class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(784, 392),
            nn.BatchNorm1d(392),
            nn.LeakyReLU(0.1),
            nn.Linear(392, 196),
            nn.BatchNorm1d(196),
            nn.LeakyReLU(0.1),
            nn.Linear(196, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, hidden_size),
        )
        self.hidden2mu = nn.Linear(hidden_size, hidden_size)
        self.hidden2log_var = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, torch.exp(log_var / 2)


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 196),
            nn.BatchNorm1d(196),
            nn.LeakyReLU(0.1),
            nn.Linear(196, 392),
            nn.BatchNorm1d(392),
            nn.LeakyReLU(0.1),
            nn.Linear(392, 784),
            Stack(1, 28, 28),
            nn.Tanh(),
        )

    def forward(self, x):
        mu = self.decoder(x)
        return mu, torch.ones_like(mu)


@click.command()
@click.argument("dataset", type=str, default="MNIST")
@click.option("--cond/--not-cond")
@click.option("--n_epochs", "-n", type=int, default=2)
@click.option("--batch_size", "-b", type=int, default=128)
@click.option("--hidden_size", "-h", type=int, default=28)
def train(dataset, cond, n_epochs, batch_size, hidden_size):
    data_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: 2 * x - 1.0)]
    )

    # Initializing Dataloader
    if dataset == "MNIST":
        dataset_klass = MNIST
    elif dataset == "FashionMNIST":
        dataset_klass = FashionMNIST
    else:
        raise click.ClickException(f"Unrecognized dataset: {dataset}")
    train_set = dataset_klass(
        "data/", download=True, train=True, transform=data_transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_set = dataset_klass(
        "data/", download=True, train=False, transform=data_transform
    )
    val_loader = DataLoader(val_set, batch_size=batch_size)

    hyper_params = {"alpha": 1, "lr": 0.005}

    encoder = Encoder(hidden_size)
    decoder = Decoder(hidden_size)
    conditioner = Cond(hidden_size)

    if cond:
        model = CVAE(
            encoder=encoder,
            decoder=decoder,
            conditioner=conditioner,
            visible_distribution=dist.Normal,
            **hyper_params,
        )
    else:
        model = VAE(
            encoder=encoder,
            decoder=decoder,
            visible_distribution=dist.Normal,
            hidden_dim=hidden_size,
            **hyper_params,
        )

    # Initializing Trainer and setting parameters
    trainer = Trainer(gpus=0, auto_lr_find=True, max_epochs=n_epochs)
    # Using trainer to fit vae model to dataset
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
