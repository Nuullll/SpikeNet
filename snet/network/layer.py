# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: layer.py
# @Last modified by:   Yilong Guo
# @Last modified time: 04-Sep-2018


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
