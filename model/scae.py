import torch
import torch.nn as nn
from model.memory import MemoryModule
from torch.nn import functional as F

class ConvDown(nn.Module):
    """1D-conv => downsample"""

    def __init__(self, in_channels, out_channels, latent=False):
        super(ConvDown, self).__init__()
        self.latent = latent
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.down = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm1d(out_channels),
            # nn.LeakyReLU(0.1)
        )
        self.bottle = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        if self.latent:
            x = self.conv(x)
            bottle = self.bottle(x)
            return bottle
        else:
            x = self.conv(x)
            down = self.down(x)
            return down

class ConvUp(nn.Module):
    """1D-conv => upsample"""

    def __init__(self, in_channels, out_channels, recon=False):
        super(ConvUp, self).__init__()
        self.recon = recon
        self.up = nn.Sequential(
            nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm1d(in_channels),
            # nn.LeakyReLU(0.1)
        )
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.recon = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        if self.recon:
            up = self.up(x)
            recon = self.recon(up)
            return recon
        else:
            up = self.up(x)
            x = self.conv(up)
            return x

class Encoder(nn.Module):
    def __init__(self, n_channels, init_nc):
        super(Encoder, self).__init__()
        self.down1 = ConvDown(n_channels, init_nc)
        self.down2 = ConvDown(init_nc, init_nc*2)
        self.down3 = ConvDown(init_nc*2, init_nc*4)
        self.down4 = ConvDown(init_nc*4, init_nc*8, latent=True)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return x4

class Decoder(nn.Module):
    def __init__(self, n_channels, init_nc):
        super(Decoder, self).__init__()
        self.up1 = ConvUp(init_nc*8, init_nc*4)
        self.up2 = ConvUp(init_nc*4, init_nc*2)
        self.up3 = ConvUp(init_nc*2, init_nc)
        self.up4 = ConvUp(init_nc, n_channels, recon=True)

    def forward(self, x):
        x1 = self.up1(x)
        x2 = self.up2(x1)
        x3 = self.up3(x2)
        x4 = self.up4(x3)
        return x4

class SCAE(nn.Module):
    def __init__(self, n_channels, init_nc):
        super(SCAE, self).__init__()
        self.encoder = Encoder(n_channels, init_nc)
        self.decoder = Decoder(n_channels, init_nc)

    def forward(self, x):
        fea = self.encoder(x)
        rec = self.decoder(fea)
        return rec, fea

class MASCAE(nn.Module):
    def __init__(self, n_channels, init_nc, mem_dim):
        super(MASCAE, self).__init__()
        self.encoder = Encoder(n_channels, init_nc)
        self.decoder = Decoder(n_channels, init_nc)
        self.memory = MemoryModule(mem_dim=mem_dim, fea_dim=init_nc*8)

    def forward(self, x):
        x_query = self.encoder(x)
        x_update, fea_loss, sep_loss = self.memory(x_query)
        rec = self.decoder(x_update)
        rec_query = self.encoder(rec)
        rec_loss = nn.MSELoss()(rec, x)
        enc_loss = nn.MSELoss()(x_query, rec_query)
        return rec_loss, fea_loss, sep_loss, enc_loss, x_query, x_update, rec
