
import numpy as np
import torch


def RRAM_device_variation(variation, value):
    return torch.tensor(np.random.normal(value, variation * value)).float()

