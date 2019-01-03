
import os
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import random


DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')


def main():

    dataset = MNIST(DATASET_DIR, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).squeeze(0))
    ]))

    while True:
        idx = random.randint(0, 60000)

        image, label = dataset[idx]
        plt.imshow(image, cmap='Greys')
        plt.title(label)
        plt.pause(3)


if __name__ == '__main__':
    main()
