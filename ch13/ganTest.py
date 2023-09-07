import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from gan import Generator, Discriminator, train_gan
import os

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("selected device is {}".format(device))


fashion_mnist_train = torchvision.datasets.FashionMNIST(root='data/FashionMNIST', train=True,\
                                                        download=True, transform=transforms.ToTensor()).data
fashion_mnist_train = torch.unsqueeze(fashion_mnist_train, 1)
fashion_mnist_train = fashion_mnist_train.to(torch.float32) / 255.0

batch_size = 128

generator = Generator().to(device)
discriminator = Discriminator().to(device)

# load model if exists, make dir if not exists
if not os.path.exists('models'):
    os.mkdir('models')
if os.path.isfile('models/generator.pt') and os.path.isfile('models/discriminator.pt'):
    generator.load_state_dict(torch.load('models/generator.pt'))
    discriminator.load_state_dict(torch.load('models/discriminator.pt'))
else:
    train_gan(generator, discriminator, fashion_mnist_train, n_epochs=10, batch_size=batch_size, zdim=100, verbose=1)

    # save model
    torch.save(generator.state_dict(), 'models/generator.pt')
    torch.save(discriminator.state_dict(), 'models/discriminator.pt')


# Test the model

generator.eval()
discriminator.eval()

n_row = 10
p_real, p_fake = 0., 0.
for steps in range(100):
    with torch.no_grad():
        z = torch.randn(n_row, 100).to(device)
        generated_data = generator(z)


# show generated images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for idx, data in enumerate(generated_data):
    data = data.cpu().numpy().squeeze(0)
    axes[idx // 5][idx % 5].imshow(data, cmap='gray')
    axes[idx // 5][idx % 5].axis('off')
plt.show()
