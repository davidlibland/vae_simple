import os
import random
from typing import Type

import pytorch_lightning as pl
import torch
import torch.distributions as dist
import torch.nn as nn
from torch.optim import Adam
from torchvision.utils import save_image


class CVAE(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        conditioner: nn.Module,
        visible_distribution: Type[dist.Distribution],
        alpha=1,
        lr=1e-3,
    ):
        """
        The encoder and conditioner should both output tuples mu, std
        describing a latent mean and standard deviation, of the same shape.

        The decoder should output a tuple of parameters for the visible distribution.

        The visible distribution will be applied to data to compute the log loss.
        """
        super().__init__()
        self.lr = lr
        self.alpha = alpha

        self.visible_distribution = visible_distribution

        self.encoder = encoder
        self.decoder = decoder
        self.conditioner = conditioner

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

    def _loss_and_output_step(self, batch):
        x, y = batch
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        # Encode the data
        mu, var = self.encoder(x)
        p = dist.Normal(mu, var)
        # p is the distribution encoding the data

        # q is the conditional distribution
        q = dist.Normal(*self.conditioner(y))
        kl_loss = dist.kl.kl_divergence(p, q).sum(dim=1).mean()
        # Compute the kl-loss of the two.

        # Use the reparametrization trick to take hidden samples:
        hidden = p.rsample()

        # Compute the reconstruction:
        reconstruction = self.visible_distribution(*self.decoder(hidden))

        # Compute the reconstruction loss:
        recon_loss = (
            -reconstruction.log_prob(x.view(x.size(0), 1, 28, 28))
            .mean(dim=0)
            .sum()
        )

        # This is the total loss:
        loss = recon_loss * self.alpha + kl_loss

        # Predictions are given by the mean:
        x_out = reconstruction.mean
        return kl_loss, loss, recon_loss, x_out

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn(size=(mu.size(0), mu.size(1)))
        z = z.type_as(mu)  # Setting z to be .cuda when using GPU training
        return mu + sigma * z

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        p = dist.Normal(*self.encoder(x))
        hidden = p.rsample()
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
