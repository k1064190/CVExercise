import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sampling(args):
    z_mean, z_log_var = args
    epsilon = torch.randn_like(z_mean)  # z_mean.shape = (batch_size, zdim)
    return z_mean + torch.exp(z_log_var / 2) * epsilon

class Encoder(nn.Module):
    def __init__(self, in_channel=1, zdim=32, size=28):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 64x14x14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 128x7x7
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),  # 256x7x7 # 256xsize/4xsize/4
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.flatten = nn.Flatten()
        self.z_mean = nn.Linear(256 * (size // 4) * (size // 4), zdim)
        self.z_log_var = nn.Linear(256 * (size // 4) * (size // 4), zdim)
        self.encoder_output = torchvision.transforms.Lambda(sampling)

    def forward(self, x):
        # data를 이용해 z_mean, z_log_var을 학습하고, 이를 이용해 gaussian 분포에서 샘플링한 z를 반환
        x = self.encoder(x)
        x = self.flatten(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.encoder_output([z_mean, z_log_var])
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    def __init__(self, out_channel=1, zdim=32, size=28):
        super().__init__()

        self.fc = nn.Linear(zdim, 256 * (size // 4) * (size // 4))
        self.unflatten = nn.Unflatten(1, (256, size // 4, size // 4))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),  # 128x7x7
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),  # 64x14x14
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),  # 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, out_channel, kernel_size=3, stride=1, padding=1, bias=False),  # 1x28x28
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class VaeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_hat, z_mean, z_log_var):
        reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='none')    # (batch_size, 1, 28, 28)
        reconstruction_loss = torch.sum(reconstruction_loss, dim=(1, 2, 3)) # (batch_size)
        reconstruction_loss = torch.mean(reconstruction_loss)           # (1)
        kl_divergence = 1 + z_log_var - z_mean.pow(2) - z_log_var.exp()    # (batch_size, zdim)
        kl_divergence = -0.5 * torch.sum(kl_divergence, dim=1)            # (batch_size)
        kl_divergence = torch.mean(kl_divergence)                         # (1)
        total_loss = reconstruction_loss + kl_divergence
        return total_loss




