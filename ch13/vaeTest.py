import random

import cv2 as cv
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from vae import Encoder, Decoder, VaeLoss

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
torch.manual_seed(0)

# Load MNIST dataset
mnist_train = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=torchvision.transforms.ToTensor())

# Create a dataloader
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=True)

zdim = 32


# if model exists, load it
import os
if os.path.isfile('models/encoder_vae.pth') and os.path.isfile('models/decoder_vae.pth'):
    encoder = Encoder(1, zdim, 28)
    decoder = Decoder(1, zdim, 28)
    encoder.to(device)
    decoder.to(device)
    encoder.load_state_dict(torch.load('models/encoder_vae.pth'))
    decoder.load_state_dict(torch.load('models/decoder_vae.pth'))
else:
    encoder = Encoder(1, zdim, 28)
    decoder = Decoder(1, zdim, 28)
    encoder.to(device)
    decoder.to(device)
    params = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optimizer = torch.optim.Adam(params, lr=0.001)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = VaeLoss()

    step_losses = []
    epoch_losses = []

    # Train the model
    num_epochs = 10
    for epoch in tqdm(range(num_epochs)):
        encoder.train()
        decoder.train()
        epoch_loss = 0
        for data, _ in tqdm(train_loader):
            data = data.to(device)
            # (batch_size, 1, 28, 28)
            mu, logvar, encoded_data = encoder(data)    # z_mean, z_log_var, z
            decoded_data = decoder(encoded_data)    # x_hat
            loss = criterion(data, decoded_data, mu, logvar)
            # loss = crossentropy(data, encoded_data) + 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # make directory if not exist
    if not os.path.isdir('models'):
        os.mkdir('models')
    torch.save(encoder.state_dict(), 'models/encoder_vae.pth')
    torch.save(decoder.state_dict(), 'models/decoder_vae.pth')
    print("model saved")


# Test the model with a sample
import matplotlib.pyplot as plt

mnist_test = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=16, shuffle=True)

encoder.eval()
decoder.eval()
with torch.no_grad():
    #   just 1 batch
    data = next(iter(test_loader))[0]
    data = data.to(device)
    # use encoder to encode data
    mu, logvar, encoded_data = encoder(data)
    # use decoder to decode encoded data
    decoded_data = decoder(encoded_data)
    # show original data and decoded data
    data = data.cpu().numpy()
    decoded_data = decoded_data.cpu().numpy()
    fig, axes = plt.subplots(2, 16, figsize=(16, 2)) # 2 rows, 16 columns
    for i in range(16):
        axes[0][i].imshow(data[i][0], cmap='gray')
        axes[1][i].imshow(decoded_data[i][0], cmap='gray')
    plt.show()


# Test the model with a sample
i = random.randint(0, len(mnist_test))
j = random.randint(0, len(mnist_test))

data1 = mnist_test[i][0].to(device)
data2 = mnist_test[j][0].to(device)

alpha = np.arange(0, 1.1, 0.1)
zz = np.zeros((len(alpha), zdim))

with torch.no_grad():
    z1 = encoder(data1)[2]
    z2 = encoder(data2)[2]

    for i in range(len(alpha)):
        zz[i] = alpha[i] * z1 + (1 - alpha[i]) * z2

    zz = torch.from_numpy(zz).float().to(device)
    decoded_data = decoder(zz)
    decoded_data = decoded_data.cpu().numpy()

    fig, axes = plt.subplots(1, len(alpha), figsize=(16, 2))
    for i in range(len(alpha)):
        axes[i].imshow(decoded_data[i][0], cmap='gray')
        axes[i].set_title(f'alpha={alpha[i]:.1f}')
    plt.title(f'Interpolation between {mnist_test[i][1]} and {mnist_test[j][1]}')
    plt.show()
