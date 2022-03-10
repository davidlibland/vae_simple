import click
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from models import VAE, CVAE


@click.command()
@click.argument("dataset", type=str, default="MNIST")
@click.option("--cond/--not-cond")
@click.option("--n_epochs", "-n", type=int, default=2)
@click.option("--batch_size", "-b", type=int, default=128)
def train(dataset, cond, n_epochs, batch_size):
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

    hyper_params = {"alpha": 100, "hidden_size": 64, "lr": 0.005}

    if cond:
        model = CVAE(**hyper_params)
    else:
        model = VAE(**hyper_params)

    # Initializing Trainer and setting parameters
    trainer = Trainer(gpus=0, auto_lr_find=True, max_epochs=n_epochs)
    # Using trainer to fit vae model to dataset
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()
