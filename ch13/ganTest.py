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
    train_gan(generator, discriminator, fashion_mnist_train, batch_size, zdim=100, verbose=1)

    # save model
    torch.save(generator.state_dict(), 'models/generator.pt')
    torch.save(discriminator.state_dict(), 'models/discriminator.pt')


# Test the model with a sample
fashion_mnist_test = torchvision.datasets.FashionMNIST(root='data/FashionMNIST', train=False,\
                                                        download=True, transform=transforms.ToTensor()).data
fashion_mnist_test_loader = torch.utils.data.DataLoader(fashion_mnist_test, batch_size=16, shuffle=True)

generator.eval()
discriminator.eval()

p_real, p_fake = 0., 0.
for data, _ in fashion_mnist_test_loader:
    with torch.no_grad():
        data = data.to(device)
        z = torch.randn(data.shape[0], 100).to(device)
        generated_data = generator(z)
        p_real += discriminator(data).mean().item()
        p_fake += discriminator(generated_data).mean().item()

p_real /= len(fashion_mnist_test_loader)
p_fake /= len(fashion_mnist_test_loader)

print(f'p_real: {p_real:.4f}, p_fake: {p_fake:.4f}')

# show generated images

