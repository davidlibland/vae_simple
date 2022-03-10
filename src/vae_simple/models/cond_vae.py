import os
import random
import pytorch_lightning as pl
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from torchvision.utils import save_image


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


class CVAE(pl.LightningModule):
    def __init__(self, alpha=1, lr=1e-3, hidden_size=28):
        # Autoencoder only requires 1 dimensional argument since input and output-size is the same
        super().__init__()
        self.hidden_size = hidden_size
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
        self.lr = lr

        self.hiddenc2mu = nn.Linear(10, hidden_size)

        self.hidden2mu = nn.Linear(hidden_size, hidden_size)
        self.hidden2log_var = nn.Linear(hidden_size, hidden_size)
        self.alpha = alpha
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

        self.recon_loss_criterion = nn.MSELoss()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=(self.lr or self.learning_rate))

    def training_step(self, batch, batch_idx):
        kl_loss, loss, recon_loss, x_out = self._loss_and_output_step(batch)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        kl_loss, loss, recon_loss, x_out = self._loss_and_output_step(batch)
        self.log("val_kl_loss", kl_loss, on_step=True, on_epoch=True)
        self.log("val_recon_loss", recon_loss, on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return x_out, loss

    def _kl_loss(self, mu0, log_var0, mu1, log_var1):
        p = dist.Normal(mu0, torch.exp(log_var0 / 2))
        q = dist.Normal(mu1, torch.exp(log_var1 / 2))
        return dist.kl.kl_divergence(p, q)

    def _loss_and_output_step(self, batch):
        x, y = batch
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        mu, log_var = self.encode(x)
        muc, log_varc = self.condition(y)
        kl_loss = self._kl_loss(mu, log_var, muc, log_varc).sum(dim=1).mean()
        hidden = self.reparametrize(mu, log_var)
        x_out = self.decode(hidden)
        recon_loss = self.recon_loss_criterion(x, x_out.view(x.size(0), -1))
        loss = recon_loss * self.alpha + kl_loss
        return kl_loss, loss, recon_loss, x_out

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn(size=(mu.size(0), mu.size(1)))
        z = z.type_as(mu)  # Setting z to be .cuda when using GPU training
        return mu + sigma * z

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var

    def condition(self, y):
        y_one_hot = F.one_hot(y, num_classes=10).float()
        mu = self.hiddenc2mu(y_one_hot)
        return mu, torch.zeros_like(mu)

    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        return self.decoder(hidden)

    def scale_image(self, img):
        out = (img + 1) / 2
        return out

    def validation_epoch_end(self, outputs):
        if not os.path.exists("vae_images"):
            os.makedirs("vae_images")
        choice = random.choice(outputs)  # Choose a random batch from outputs
        output_sample = choice[0]  # Take the recreated image
        output_sample = output_sample.reshape(
            -1, 1, 28, 28
        )  # Reshape tensor to stack the images nicely
        save_image(
            output_sample, f"vae_images/epoch_{self.current_epoch+1}.png"
        )
