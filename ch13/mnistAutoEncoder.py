import cv2 as cv
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from autoEncoder import Encoder, Decoder

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

# Load MNIST dataset
mnist_train = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=torchvision.transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Create a dataloader
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=128, shuffle=False)

# Print the shape of the dataset

zdim = 32

# # make encoder
# encoder = torch.nn.Sequential()
# encoder.add_module('conv1', torch.nn.Conv2d(1, 32, kernel_size=3, padding=1))
# encoder.add_module('bn1', torch.nn.BatchNorm2d(32))
# encoder.add_module('relu1', torch.nn.ReLU())
# encoder.add_module('conv2', torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2))
# encoder.add_module('bn2', torch.nn.BatchNorm2d(64))
# encoder.add_module('relu2', torch.nn.ReLU())
# encoder.add_module('conv3', torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2))
# encoder.add_module('bn3', torch.nn.BatchNorm2d(64))
# encoder.add_module('relu3', torch.nn.ReLU())
# encoder.add_module('conv4', torch.nn.Conv2d(64, 64, kernel_size=3, padding=1))
# encoder.add_module('bn4', torch.nn.BatchNorm2d(64))
# encoder.add_module('relu4', torch.nn.ReLU())
# encoder.add_module('flatten', torch.nn.Flatten())
# encoder.add_module('fc1', torch.nn.Linear(7*7*64, 128))
# encoder.add_module('relu5', torch.nn.ReLU())
# encoder.add_module('fc2', torch.nn.Linear(128, zdim))
#
# # make decoder
# decoder = torch.nn.Sequential()
# decoder.add_module('fc1', torch.nn.Linear(zdim, 128))
# decoder.add_module('relu1', torch.nn.ReLU())
# decoder.add_module('fc2', torch.nn.Linear(128, 7*7*64))
# decoder.add_module('relu2', torch.nn.ReLU())
# decoder.add_module('reshape', torch.nn.Unflatten(1, (64, 7, 7)))
# decoder.add_module('conv1', torch.nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1))
# decoder.add_module('bn1', torch.nn.BatchNorm2d(64))
# decoder.add_module('relu3', torch.nn.ReLU())
# decoder.add_module('conv2', torch.nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1))
# decoder.add_module('bn2', torch.nn.BatchNorm2d(64))
# decoder.add_module('relu4', torch.nn.ReLU())
# decoder.add_module('conv3', torch.nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1))
# decoder.add_module('bn3', torch.nn.BatchNorm2d(32))
# decoder.add_module('relu5', torch.nn.ReLU())
# decoder.add_module('conv4', torch.nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1))
#
# model = torch.nn.Sequential(encoder, decoder)
# model = model.to(device)

encoder = Encoder(zdim).to(device)
decoder = Decoder(zdim).to(device)
params = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

optimizer = torch.optim.Adam(params, lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

step_losses = []
epoch_losses = []

# Train the model
num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    encoder.train()
    decoder.train()
    epoch_loss = 0
    for data, _ in tqdm(train_loader, total=len(train_loader), leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        encoded_data = encoder(data)
        decoded_data = decoder(encoded_data)
        loss = criterion(decoded_data, data)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_losses.append(loss.item())
    epoch_losses.append(epoch_loss / len(train_loader))
    print('epoch: {}, loss: {:.4f}'.format(epoch, epoch_loss))


# Plot the losses
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].plot(step_losses)
axs[1].plot(epoch_losses)
plt.show()


exit()
# Generate samples with the test data
import random

i = random.randint(0, len(mnist_test))
j = random.randint(0, len(mnist_test))
x = np.array((mnist_test[i][0], mnist_test[j][0]))
x = torch.from_numpy(x).to(device)


