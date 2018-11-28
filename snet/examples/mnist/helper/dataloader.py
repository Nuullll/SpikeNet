# @Author: Yilong Guo
# @Date:   28-Oct-2018
# @Email:  vfirst218@gmail.com
# @Filename: analysis.py
# @Last modified by:   Yilong Guo
# @Last modified time: 28-Oct-2018


import os
import torch
import wget
import platform
from mnist import MNIST


class MNISTLoader(object):
    """
    MNIST dataset loader.
    """
    def __init__(self, mndir=None, downloaded=True):
        if mndir is None:
            if platform.node() == "Nuullll-MBP.local":
                mndir = '/Users/nuullll/Projects/SNN-AutoEncoder/dataset/mnist'
            elif platform.node() == 'Nuullll-Lab-win10':
                mndir = 'E:\Projects\SNN-AutoEncoder\dataset\mnist'
            else:
                mndir = os.path.join(os.path.dirname(__file__), '../dataset')

        if not downloaded:
            # Prepare MNIST dataset
            if not os.path.exists(mndir):
                os.makedirs(mndir)

            files = [
                'train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz'
            ]

            # Download raw data
            for file in files:
                url = 'http://yann.lecun.com/exdb/mnist/' + file
                wget.download(url=url, out=mndir)

        # Parse MNIST
        mndata = MNIST(mndir)
        mndata.gz = True

        self.data = mndata
        self.images = None
        self.labels = None

    def load_training(self):
        images, labels = self.data.load_training()

        self.images = torch.tensor(images)
        self.labels = torch.tensor(labels)

        return self.images, self.labels

    def load_testing(self):
        images, labels = self.data.load_testing()

        return torch.tensor(images), torch.tensor(labels)

    def downsample(self, target_size, images=None):
        """
        Downsamples self.images.
        :param target_size:     (output_height, output_weight)
        :param images:          the images to be downsampled.
        :return: (batch_size, output_height * output_weight)
        """
        if images is None:
            images = self.images

        images = images.view(-1, 28, 28).float()

        images = torch.nn.functional.interpolate(images.unsqueeze(1), target_size).squeeze(1).long()\
            .view(images.shape[0], -1)

        return images
