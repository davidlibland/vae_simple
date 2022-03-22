import random
from typing import Type

import pytorch_lightning as pl
import torch.distributions as dist
import torch.nn as nn
from torch.optim import Adam


class CVAE(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        conditioner: nn.Module,
        visible_distribution: Type[dist.Distribution],
        plot_samples=None,
        plot_reconstructions=None,
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
        self.plot_samples = plot_samples
        self.plot_reconstructions = plot_reconstructions

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
        # Encode the data
        mu, var = self.encoder(y)
        p = dist.Normal(mu, var)
        # p is the distribution encoding the data

        # q is the conditional distribution
        q = dist.Normal(*self.conditioner(x))
        kl_loss = dist.kl.kl_divergence(p, q).sum(dim=1).mean()
        # Compute the kl-loss of the two.

        # Use the reparametrization trick to take hidden samples:
        hidden = p.rsample()

        # Compute the reconstruction:
        reconstruction = self.visible_distribution(*self.decoder(hidden))

        # Compute the reconstruction loss:
        recon_loss = -reconstruction.log_prob(y).mean(dim=0).sum()

        # This is the total loss:
        loss = recon_loss * self.alpha + kl_loss

        # Predictions are given by the mean:
        y_out = reconstruction.mean
        return kl_loss, loss, recon_loss, y_out

    def forward(self, x):
        p = dist.Normal(*self.encoder(x))
        hidden = p.rsample()
        reconstruction = self.visible_distribution(*self.decoder(hidden))
        return reconstruction.mean

    def sample(self, x):
        q = dist.Normal(*self.conditioner(x))
        hidden = q.sample()
        reconstruction = self.visible_distribution(*self.decoder(hidden))
        return reconstruction.mean

    def validation_epoch_end(self, outputs):
        choice = random.choice(outputs)  # Choose a random batch from outputs
        output_sample = choice[0]  # Take the recreated image
        epoch = self.current_epoch
        if self.plot_reconstructions:
            self.plot_reconstructions(output_sample, epoch)
        if self.plot_samples:
            self.plot_samples(self, epoch)
