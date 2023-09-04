import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

# AutoEncoder for MNIST

class Encoder(nn.Module):
    def __init__(self, zdim=32):
        super().__init__()

        # Convolution
        # self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False) # 8x28x28
        # self.bn1 = nn.BatchNorm2d(8)
        # self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False) # 16x14x14
        # self.bn2 = nn.BatchNorm2d(16)
        # self.relu2 = nn.ReLU()
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False) # 32x7x7
        # self.bn3 = nn.BatchNorm2d(32)
        # self.relu3 = nn.ReLU()
        # self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False) # 64x7x7
        # self.bn4 = nn.BatchNorm2d(64)
        # self.relu4 = nn.ReLU()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False), # 8x28x28
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False), # 16x14x14
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False), # 32x7x7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False), # 64x7x7
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Flatten
        self.flatten = nn.Flatten()

        # fc layer
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, zdim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, zdim=32):
        super().__init__()

        # fc layer
        self.fc = nn.Sequential(
            nn.Linear(zdim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64*7*7),
            nn.ReLU()
        )

        # unflatten
        self.unflatten = nn.Unflatten(1, (64, 7, 7))

        # conv transpose
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False), # 32x7x7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # 16x14x14
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False), # 8x28x28
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1, bias=False), # 1x28x28
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.decoder_cnn(x)
        return x



class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

    def encode(self, x):
        z = self.encoder(x)
        return z

    def decode(self, z):
        x = self.decoder(z)
        return x





