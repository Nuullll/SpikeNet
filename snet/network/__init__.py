# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: __init__.py
# @Last modified by:   Yilong Guo
# @Last modified time: 04-Sep-2018


import torch
from .layer import *
from .connection import *
from .monitor import *
from ..settings import *

import matplotlib.pyplot as plt


class NetworkLoader(object):
    """
    Load network from settings file.
    """
    def load_default(self, weight_map=None):
        # Specify sizes
        sizes = {
            'I': NetworkConfig.INPUT_NEURON_NUMBER,
            'O': NetworkConfig.OUTPUT_NEURON_NUMBER
        }

        # Specify layers
        # {name: <Layer>}
        layers = {
            'I': PoissonLayer(firing_step=torch.zeros(sizes['I'])),
            'O': LIFLayer(sizes['O'])
        }

        # Specify weights
        # {(source, target): weight}
        # will be converted into
        # {(source, target): <Connection>}
        if weight_map is None:
            weights = {
                ('I', 'O'): ConnectionConfig.W_MIN + (ConnectionConfig.W_MAX - ConnectionConfig.W_MIN) * torch.rand(
                    sizes['I'], sizes['O'])
            }
        else:
            weights = {
                ('I', 'O'): weight_map
            }

        # Specify monitors (monitors will be instantiated later during the building process of network)
        # {name: [state_vars]}
        monitors = {
            'I': ['o'],
            'O': ['o']
        }

        # Config network
        builder = NetworkBuilder(layers=layers, weights=weights, monitors=monitors)
        network = builder.build()

        return network


class NetworkBuilder:
    """
    Network builder, with given <Layer>s, <torch.Tensor> weights, <Monitor>s.
    """
    def __init__(self, layers, weights, monitors):
        """
        Initialize builder.
        :param layers:      dict        {name: <Layer>}
        :param weights:     dict        {(pre_layer_name, post_layer_name): weights}
        :param monitors:    dict        {name: [state_vars]}
        """
        self.network = Network()

        # initialize layers
        self.network.layers = layers

        # initialize connections
        self._init_connections(weights)

        # initialize monitors
        self._init_monitors(monitors)

        # default mode: training
        self.network.training_mode()

    def _init_connections(self, weights):
        """
        Instantiate <Connection> objects.
        :param weights:     dict        {(pre_layer_name, post_layer_name): weights}
        """
        for key, weight in weights.items():
            pre_layer_name, post_layer_name = key
            pre_layer = self.network.layers[pre_layer_name]
            post_layer = self.network.layers[post_layer_name]
            conn = Connection(pre_layer=pre_layer, post_layer=post_layer, weight=weight)
            self.network.connections[(pre_layer_name, post_layer_name)] = conn

    def _init_monitors(self, monitors):
        """
        Instantiate <Monitor> objects.
        :param monitors:    dict        {name: [state_vars]}
        """
        for name, state_vars in monitors.items():
            mon = Monitor(target=self.network.layers[name], state_vars=state_vars)
            self.network.monitors[name] = mon

    def build(self, training=True):
        """
        :return:    <Network>
        """
        if training:
            self.network.training_mode()
        else:
            self.network.inference_mode()

        return self.network


class Network:
    """
    Responsible for interaction simulation among neurons and synapses.
    """
    def __init__(self):
        """
        Initialize <Network> instance, with a configuration.
        """
        self.layers = {}            # {name: <Layer>}
        self.connections = {}       # {(pre_layer_name, post_layer_name): <Connection>}
        self.monitors = {}          # {layer_name: <Monitor>}

        self.dt = NetworkConfig.DT_MS

        self.time = 0.

    def run(self, time):
        """
        Run simulation for given `time`.
        :param time:        float       Simulation time. (NOT steps)
        """
        # Total simulation steps
        steps = int(time / self.dt)

        # Do simulation
        for t in range(steps):
            # Update monitors
            self._update_monitors()

            # STDP updates according to incoming new pre-spikes
            self._update_on_pre_spikes(self.time + t)

            # Feed forward
            self._feed_forward()

            # Layers process
            self._process()

            # STDP updates according to incoming new post-spikes
            self._update_on_post_spikes(self.time + t)

            # if not t % 200:
            #     # Display weight map
            #     synapse = self.connections[('I', 'O')]
            #     plt.imshow(synapse.weight.numpy(), cmap='Purples', vmin=synapse.w_min, vmax=synapse.w_max, aspect='auto')
            #     plt.pause(0.00001)

        self.time += time

    def after_batch(self):
        """
        Updates network after one batch, e.g. adapts thresholds.
        """
        for lyr in self.layers.values():
            lyr.adapt_thresholds()
            lyr.clear_spike_counts()

    def _update_monitors(self):
        """
        Feeds state values to monitors.
        """
        for mon in self.monitors.values():
            mon.update()

    def _update_on_pre_spikes(self, time):
        """
        Updates weights when new pre-spikes come.
        """
        for conn in self.connections.values():
            conn.update_on_pre_spikes(time)

    def _feed_forward(self):
        """
        Feeds pre-synaptic signals to synapses, and then pass results to post-synaptic neurons' input.
        """
        for conn in self.connections.values():
            conn.feed_forward()

    def _update_on_post_spikes(self, time):
        """
        Updates weights when new post-spikes come.
        """
        for conn in self.connections.values():
            conn.update_on_post_spikes(time)

    def _process(self):
        """
        Each layer processes. Generates new output for each layer.
        """
        for lyr in self.layers.values():
            lyr.process()

    def training_mode(self):
        """
        Turns on training mode.
        """
        for lyr in self.layers.values():
            lyr.adaptive = True
            lyr.inhibition = True

        for conn in self.connections.values():
            conn.static = False

    def inference_mode(self):
        """
        Turns on inference mode.
        """
        for lyr in self.layers.values():
            lyr.adaptive = False
            # lyr.inhibition = False

        for conn in self.connections.values():
            conn.static = True
