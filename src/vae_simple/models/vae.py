import random

import pytorch_lightning as pl
import torch
import torch.distributions as dist
from torch.optim import Adam


class VAE(pl.LightningModule):
    def __init__(
        self,
        encoder,
        decoder,
        visible_distribution,
        hidden_dim,
        plot_samples=None,
        plot_reconstructions=None,
        alpha=1,
        lr=0.05,
    ):
        # Autoencoder only requires 1 dimensional argument since input and output-size is the same
        super().__init__()
        self.lr = lr
        self.alpha = alpha

        self.visible_distribution = visible_distribution

        self.encoder = encoder
        self.decoder = decoder

        self.hidden_dim = hidden_dim
        self.plot_samples = plot_samples
        self.plot_reconstructions = plot_reconstructions

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
        x = batch
        # Encode the data
        mu, var = self.encoder(x)
        p = dist.Normal(mu, var)
        # p is the distribution encoding the data

        # q is the conditional distribution
        q = dist.Normal(torch.zeros_like(mu), torch.ones_like(mu))
        kl_loss = dist.kl.kl_divergence(p, q).sum(dim=1).mean()
        # Compute the kl-loss of the two.

        # Use the reparametrization trick to take hidden samples:
        hidden = p.rsample()

        # Compute the reconstruction:
        reconstruction = self.visible_distribution(*self.decoder(hidden))

        # Compute the reconstruction loss:
        recon_loss = -reconstruction.log_prob(x).mean(dim=0).sum()

        # This is the total loss:
        loss = recon_loss * self.alpha + kl_loss

        # Predictions are given by the mean:
        x_out = reconstruction.mean
        return kl_loss, loss, recon_loss, x_out

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        p = dist.Normal(*self.encoder(x))
        hidden = p.rsample()
        reconstruction = self.visible_distribution(*self.decoder(hidden))
        return reconstruction.mean

    def sample(self, n):
        hidden = dist.Normal(0, 1).sample((n, self.hidden_dim))
        reconstruction = self.visible_distribution(*self.decoder(hidden))
        return reconstruction.mean

    def validation_epoch_end(self, outputs):
        choice = random.choice(outputs)  # Choose a random batch from outputs
        output_sample = choice[0]  # Take the recreated image
        epoch = self.current_epoch + 1
        if self.plot_reconstructions:
            self.plot_reconstructions(output_sample, epoch)
        if self.plot_samples:
            self.plot_samples(self, epoch)
