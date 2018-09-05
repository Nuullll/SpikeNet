# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: layer.py
# @Last modified by:   Yilong Guo
# @Last modified time: 04-Sep-2018


import torch
from .neuron import *


class Layer:
    """
    <Layer> contains a list of <Neuron>s.
    """
    neuron_map = {
        'Poisson': PoissonNeuron,
        'LIF': LIFNeuron
    }

    def __init__(self, neuron_type, size, **kwargs):
        """
        Initialize a list of neurons of specific type as a <Layer>.
        :param neuron_type:         str         'Poisson' or 'LIF'.
        :param size:                int         The number of neurons in the <Layer>.
        :param kwargs:              dict        Parameters for instantiating <Neuron>s.
        """
        self.neurons = [self.neuron_map[neuron_type](**kwargs) for i in range(size)]

    def get_state(self, state):
        """
        Gets the specific current `state` value of all neurons, as a <torch.Tensor>.
        :param state:       str
        :return:            <torch.Tensor>
        """
        return torch.tensor([getattr(neuron, state, None) for neuron in self.neurons])

    def set_state(self, state, value):
        """
        Sets the specific `state` value of all neurons.
        :param state:       str
        :param value:       <torch.Tensor>
        """
        for i, neuron in enumerate(self.neurons):
            setattr(neuron, state, value[i])

    def process(self):
        """
        Processes one time step on neurons.
        """
        for neuron in self.neurons:
            neuron.process()
