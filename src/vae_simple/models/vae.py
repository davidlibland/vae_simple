import os
import random
import pytorch_lightning as pl
from torch.optim import Adam
import torch
import torch.nn as nn

from torchvision.utils import save_image


class VAE(pl.LightningModule):
    def __init__(self, alpha=1):
        # Autoencoder only requires 1 dimensional argument since input and output-size is the same
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 196),
            nn.ReLU(),
            nn.BatchNorm1d(196, momentum=0.7),
            nn.Linear(196, 49),
            nn.ReLU(),
            nn.BatchNorm1d(49, momentum=0.7),
            nn.Linear(49, 28),
            nn.LeakyReLU(),
        )
        self.hidden2mu = nn.Linear(28, 28)
        self.hidden2log_var = nn.Linear(28, 28)
        self.alpha = alpha
        self.decoder = nn.Sequential(
            nn.Linear(28, 49),
            nn.ReLU(),
            nn.Linear(49, 196),
            nn.ReLU(),
            nn.Linear(196, 784),
            nn.Tanh(),
        )

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

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

    def _loss_and_output_step(self, batch):
        x, _ = batch
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        mu, log_var = self.encode(x)
        kl_loss = (
            -0.5 * (1 + log_var - mu**2 - torch.exp(log_var)).sum(dim=1)
        ).mean(dim=0)
        hidden = self.reparametrize(mu, log_var)
        x_out = self.decode(hidden)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
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
        output_sample = self.scale_image(output_sample)
        save_image(
            output_sample, f"vae_images/epoch_{self.current_epoch+1}.png"
        )
