import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_channels=256, latent_channels=32):
        super(VAE, self).__init__()
        self.latent_channels = latent_channels

        # Encoder
        self.encoder_conv1 = nn.Conv3d(input_channels, 128, kernel_size=3, stride=2, padding=1)
        self.encoder_gn1 = nn.GroupNorm(32, 128)
        self.encoder_conv2 = nn.Conv3d(128, 64, kernel_size=3, stride=2, padding=1)
        self.encoder_gn2 = nn.GroupNorm(32, 64)
        self.encoder_fc_mu_logvar = nn.Conv3d(64, latent_channels * 2, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.decoder_conv1 = nn.Conv3d(latent_channels, 64, kernel_size=3, stride=1, padding=1)
        self.decoder_gn1 = nn.GroupNorm(32, 64)
        self.decoder_tconv1 = nn.ConvTranspose3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.decoder_gn2 = nn.GroupNorm(32, 128)
        self.decoder_tconv2 = nn.ConvTranspose3d(128, input_channels, kernel_size=4, stride=2, padding=1)

        self.silu = nn.SiLU()

    def encode(self, x):
        x = self.silu(self.encoder_gn1(self.encoder_conv1(x)))
        x = self.silu(self.encoder_gn2(self.encoder_conv2(x)))
        mu_logvar = self.encoder_fc_mu_logvar(x)
        mu = mu_logvar[:, :self.latent_channels, :, :, :]
        log_var = mu_logvar[:, self.latent_channels:, :, :, :]
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.silu(self.decoder_gn1(self.decoder_conv1(z)))
        x = self.silu(self.decoder_gn2(self.decoder_tconv1(x)))
        x_recon = self.decoder_tconv2(x) # Output logits
        return x_recon

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
