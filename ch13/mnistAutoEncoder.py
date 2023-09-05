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

# Create a dataloader
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=True)

# Print the shape of the dataset

zdim = 32

# if model exists, load it
import os
if os.path.isfile('encoder.pth') and os.path.isfile('decoder.pth'):
    encoder = Encoder(zdim)
    decoder = Decoder(zdim)
    encoder.to(device)
    decoder.to(device)
    encoder.load_state_dict(torch.load('encoder.pth'))
    decoder.load_state_dict(torch.load('decoder.pth'))

else:
    encoder = Encoder(zdim)
    decoder = Decoder(zdim)
    encoder.to(device)
    decoder.to(device)
    params = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optimizer = torch.optim.Adam(params, lr=0.001)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()

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
            encoded_data = encoder(data)
            decoded_data = decoder(encoded_data)
            loss = criterion(decoded_data, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step_losses.append(loss.item())
        epoch_losses.append(epoch_loss / len(train_loader))
        print('epoch: {}, loss: {:.4f}'.format(epoch, epoch_losses[-1]))

    # Save the model
    torch.save(encoder.state_dict(), 'encoder.pth')
    torch.save(decoder.state_dict(), 'decoder.pth')


    # Plot the losses
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(step_losses)
    axs[1].plot(epoch_losses)
    plt.show()

# summary of the model
from torchsummary import summary
summary(encoder, (1, 28, 28))
summary(decoder, (32,))
print(encoder)
print(decoder)

# Generate samples with the test data
import torchvision
import torch
import numpy as np
import random

mnist_test = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=128, shuffle=False)

i = random.randint(0, len(mnist_test))
j = random.randint(0, len(mnist_test))

# Stack two images
data1, _ = mnist_test[i]
data2, _ = mnist_test[j]
data = torch.stack([data1, data2], dim=0)
print(data.shape)
data = data.to(device)

# Generate z by the encoder
encoder.eval()
with torch.no_grad():
    z = encoder(data)
z = z.cpu().detach().numpy()

# Generate range of z by linear interpolation
zz = np.zeros((11, zdim))
print(zz.shape) # (11, zdim)
alpha = np.arange(0, 1.1, 0.1)
for i in range(11):
    zz[i] = alpha[i] * z[0] + (1 - alpha[i]) * z[1]

zz = zz.astype(np.float32)
zz = torch.tensor(zz).to(device)

# Generate samples by the decoder
decoder.eval()
with torch.no_grad():
    samples = decoder(zz)

# Plot the generated samples
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 11, figsize=(15, 1))
for i in range(11):
    axs[i].imshow(samples[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
    axs[i].axis('off')
plt.show()

