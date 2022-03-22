import os
import random

import click
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from models import CVAE, VAE
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.utils import save_image


class SwapXY(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return y, x


class JustX(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        return x


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


def scale_image(img):
    out = (img + 1) / 2
    return out


def plot_reconstructions(output_sample, epoch):
    if not os.path.exists("vae_images"):
        os.makedirs("vae_images")
    output_sample = output_sample.reshape(
        -1, 1, 28, 28
    )  # Reshape tensor to stack the images nicely
    output_sample = scale_image(output_sample)
    save_image(output_sample, f"vae_images/epoch_{epoch+1}.png")


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
    val_set = dataset_klass(
        "data/", download=True, train=False, transform=data_transform
    )

    hyper_params = {"alpha": 1, "lr": 0.005}

    encoder = Encoder(hidden_size)
    decoder = Decoder(hidden_size)
    conditioner = Cond(hidden_size)

    if cond:

        def plot_samples(model, epoch):
            output_labels = torch.tensor(random.choices(range(10), k=64))
            model.eval()
            with torch.no_grad():
                # compute stuff here
                output_sample = model.sample(output_labels).reshape(
                    -1, 1, 28, 28
                )  # Reshape tensor to stack the images nicely
            model.train()

            output_sample = scale_image(output_sample)
            save_image(output_sample, f"vae_images/samples_{epoch + 1}.png")

        model = CVAE(
            encoder=encoder,
            decoder=decoder,
            conditioner=conditioner,
            visible_distribution=dist.Normal,
            plot_reconstructions=plot_reconstructions,
            plot_samples=plot_samples,
            **hyper_params,
        )
        train_loader = DataLoader(SwapXY(train_set), batch_size=batch_size)
        val_loader = DataLoader(SwapXY(val_set), batch_size=batch_size)
    else:

        def plot_samples(model, epoch):
            model.eval()
            with torch.no_grad():
                # compute stuff here
                output_sample = model.sample(64).reshape(
                    -1, 1, 28, 28
                )  # Reshape tensor to stack the images nicely
            model.train()

            output_sample = scale_image(output_sample)
            save_image(output_sample, f"vae_images/samples_{epoch + 1}.png")

        model = VAE(
            encoder=encoder,
            decoder=decoder,
            visible_distribution=dist.Normal,
            hidden_dim=hidden_size,
            plot_reconstructions=plot_reconstructions,
            plot_samples=plot_samples,
            **hyper_params,
        )
        train_loader = DataLoader(JustX(train_set), batch_size=batch_size)
        val_loader = DataLoader(JustX(val_set), batch_size=batch_size)

    # Initializing Trainer and setting parameters
    trainer = Trainer(gpus=0, auto_lr_find=True, max_epochs=n_epochs)
    # Using trainer to fit vae model to dataset
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
