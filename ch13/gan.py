import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("selected device is {}".format(device))


class Generator(nn.Module):
    def __init__(self, shape=(1, 28, 28), zdim=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(zdim, 256 * (shape[1] // 4) * (shape[2] // 4)),
            nn.BatchNorm1d(256 * (shape[1] // 4) * (shape[2] // 4)),
            nn.LeakyReLU(0.2, True)
        )
        self.unflatten = nn.Unflatten(1, (256, shape[1] // 4, shape[2] // 4))
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),  # 128x14x14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),  # 64x28x28
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, shape[0], kernel_size=3, stride=1, padding=1, bias=False),   # 1x28x28
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = self.unflatten(x)
        x = self.generator(x)
        return x



class Discriminator(nn.Module):
    def __init__(self, shape=(1, 28, 28)):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False),  # 64x28x28
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 128x14x14
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 256x7x7
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256 * (shape[1] // 4) * (shape[2] // 4), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.discriminator(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def generate_latent_points(zdim=100, n_samples=10):
    z = torch.randn(n_samples, zdim)
    return z.to(device)

def train_gan(generator, discriminator, dataset, n_epochs=10, batch_size=128, zdim=100, verbose=0):

    dataset = dataset.to(device)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0001)

    n_batches = dataset.shape[0] // batch_size
    # criterion = nn.BCELoss()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(n_epochs)):
        generator.train()
        discriminator.train()

        epoch_loss_d = 0
        epoch_loss_g = 0

        for i, x_real in tqdm(enumerate(dataloader)):
            # train discriminator
            optimizer_d.zero_grad()

            # train with real samples
            y_real = torch.ones((x_real.shape[0], 1)).to(device)
            y_real_pred = discriminator(x_real)

            # fake samples
            z = generate_latent_points(zdim, x_real.shape[0])
            x_fake = generator(z)
            y_fake = torch.zeros((x_fake.shape[0], 1)).to(device)
            y_fake_pred = discriminator(x_fake)

            # loss by paper
            loss_real = -1 * torch.log(y_real_pred)
            loss_fake = -1 * torch.log(1 - y_fake_pred)
            loss_d = torch.mean(loss_real + loss_fake)

            # # loss by cross entropy
            # loss_real = criterion(y_real_pred, y_real)
            # loss_fake = criterion(y_fake_pred, y_fake)
            # loss_d = loss_real + loss_fake

            loss_d.backward()
            optimizer_d.step()

            # train generator
            optimizer_g.zero_grad()

            # fake samples
            z = generate_latent_points(zdim, x_real.shape[0])
            x_fake = generator(z)
            y_fake_pred = discriminator(x_fake)

            # loss by paper
            loss_g = -1 * torch.log(y_fake_pred)
            loss_g = torch.mean(loss_g)

            # # loss by cross entropy
            # loss_g = criterion(y_fake_pred, y_fake)

            loss_g.backward()
            optimizer_g.step()

            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()

        epoch_loss_d /= n_batches
        epoch_loss_g /= n_batches

        print(f'Epoch {epoch+1}/{n_epochs} | Loss D: {epoch_loss_d:.4f} | Loss G: {epoch_loss_g:.4f}')

        if verbose == 1:
            # plot samples if epoch is multiple of 5
            if epoch % (n_epochs // 10) == 0:
                generator.eval()
                discriminator.eval()
                z = generate_latent_points(zdim, 10)
                x_fake = generator(z)
                fig, axes = plt.subplots(1, 10, figsize=(20, 2))
                for i in range(10):
                    axes[i].imshow(x_fake[i].detach().cpu().permute(1, 2, 0), cmap='gray')
                    axes[i].axis('off')
                plt.title(f'Epoch {epoch+1}')
                plt.show()

