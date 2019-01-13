
import os
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import random
from localconfig import LocalConfig
import snet
import torch
import logging
logging.basicConfig(level=logging.WARNING)

try:
    import telemessage
    notification = True
except ImportError:
    notification = False


DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def load_bimnist(train=True):

    dataset = MNIST(DATASET_DIR, train=train, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: (x > 0.5).squeeze(0))
    ]))

    return dataset


def load_config():
    return LocalConfig(os.path.join(os.path.dirname(__file__), 'network.ini'))


def cfg_abstract(cfg, prefix='', suffix=''):
    """
    :return:    Config abstract string.
    """
    if not prefix == '':
        prefix = prefix + '.'
    if not suffix == '':
        suffix = '.' + suffix

    return f"{prefix}{cfg.network.output_neuron_number}out." \
           f"{cfg.lif_layer.winners}winners." \
           f"{cfg.input.pattern_firing_rate}pattern." \
           f"{cfg.input.background_firing_rate}background" \
           f"{suffix}"


def train(training_dataset, cfg, overwrite_check=True):
    # folder = os.path.join(RESULTS_DIR, cfg_abstract(cfg))
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # else:
    #     if overwrite_check:
    #         logging.warning('Result folder %s already exists, press enter to overwrite history result.' % folder)
    #         input()
    #
    # if notification:
    #     # send message
    #     telemessage.notify_me('Training job started.')

    # build network
    network = snet.NetworkLoader().from_cfg(config=cfg)
    network.training_mode()

    for image, label in training_dataset:
        if label not in [0, 1, 2]:
            continue
        # input image
        network.input_image(image)

        # run simulation
        network.run(cfg.input.duration_per_training_image)
        # plt.plot(network.monitors['O'].record['v'].numpy())
        # plt.show()
        print(label)
        print(network.layers['O'].spike_counts)
        print(network.layers['O'].v_th)
        network.after_batch()

    # # save final weight
    # weight_file = os.path.join(folder, 'final_weight.pt')
    torch.save(network.connections[('I', 'O')].weight, 'final_weight.pt')
    torch.save(network.layers['O'].v_th, 'v_th.pt')
    #
    # # save training config
    # cfg.save(os.path.join(folder, 'training.ini'))
    #
    # if notification:
    #     telemessage.notify_me('Training job completed.')
    #
    # return folder


if __name__ == '__main__':
    config = load_config()
    ds = load_bimnist()
    train(ds, config)
