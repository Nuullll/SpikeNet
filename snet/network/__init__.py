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
from localconfig import LocalConfig
import os.path

import matplotlib.pyplot as plt


class NetworkLoader(object):

    def from_cfg(self, config=None, weight_map=None):
        """
        Loads network parameters from <configparser.ConfigParser> object.
        :param config:      <LocalConfig>
        :param weight_map:  <torch.tensor>
        :return:            <Network>
        """
        if config is None:
            default_path = os.path.dirname(__file__)
            config = LocalConfig(os.path.join(default_path, 'template.ini'))

        config = self._infer_fields(config)

        # Specify sizes
        sizes = {
            'I': config.network.input_neuron_number,
            'O': config.network.output_neuron_number
        }

        # Specify layers
        # {name: <Layer>}
        layers = {
            'I': PoissonLayer(config=config),
            'O': LIFLayer(sizes['O'], config=config)
        }

        # Specify weights
        # {(source, target): weight}
        # will be converted into
        # {(source, target): <Connection>}
        if weight_map is None:
            if config.synapse.init_weights == 'min':
                weight_map = config.synapse.w_min * torch.ones(sizes['I'], sizes['O'])
            elif config.synapse.init_weights == 'max':
                weight_map = config.synapse.w_max * torch.ones(sizes['I'], sizes['O'])
            elif config.synapse.init_weights == 'random':
                weight_map = config.synapse.w_min + (config.synapse.w_max - config.synapse.w_min) \
                    * torch.rand(sizes['I'], sizes['O'])
            else:
                raise ValueError('Wrong configuration for synapse.init_weights')

        weights = {
            ('I', 'O'): weight_map
        }

        # Specify monitors (monitors will be instantiated later during the building process of network)
        # {name: [state_vars]}
        monitors = {
            # 'I': ['o'],
            'O': ['v']
        }

        # Config network
        builder = NetworkBuilder(layers=layers, weights=weights, monitors=monitors, config=config)
        network = builder.build()

        return network

    def _infer_fields(self, config):
        """
        Infers additional fields in config.
        :param config:      <LocalConfig>
        :return:            <LocalConfig>
        """
        # infer additional fields
        config.network.dt_s = config.network.dt_ms / 1000
        config.network.input_neuron_number = config.input.image_width * config.input.image_height
        config.synapse.learn_rate_p_eff = config.synapse.learn_rate_p * config.synapse.learn_rate_p_scaling
        config.synapse.learn_rate_m_eff = config.synapse.learn_rate_m * config.synapse.learn_rate_m_scaling
        config.synapse.decay_eff = config.synapse.decay * config.synapse.decay_scaling
        config.lif_layer.activated_phase_target = config.lif_layer.track_phase / config.network.output_neuron_number

        return config


class NetworkBuilder:
    """
    Network builder, with given <Layer>s, <torch.Tensor> weights, <Monitor>s.
    """
    def __init__(self, layers, weights, monitors, config):
        """
        Initialize builder.
        :param layers:      dict        {name: <Layer>}
        :param weights:     dict        {(pre_layer_name, post_layer_name): weights}
        :param monitors:    dict        {name: [state_vars]}
        """
        self.network = Network(config)

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
            conn = Connection(pre_layer=pre_layer, post_layer=post_layer, weight=weight, config=self.network.config)
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
    def __init__(self, config):
        """
        Initialize <Network> instance, with a configuration.
        """
        self.layers = {}            # {name: <Layer>}
        self.connections = {}       # {(pre_layer_name, post_layer_name): <Connection>}
        self.monitors = {}          # {layer_name: <Monitor>}

        self.time = 0.

        self.config = config

        # for performance
        self.dt_ms = self.config.network.dt_ms
        self.dt_s = self.dt_ms / 1000
        self.input_firing_rate = self.config.input.average_firing_rate
        self.input_neuron_number = self.config.network.input_neuron_number

        self.image = None

    def run(self, time):
        """
        Run simulation for given `time`.
        :param time:        float       Simulation time. (NOT steps)
        """
        # Total simulation steps
        steps = int(time / self.dt_ms)

        # Do simulation
        for t in range(steps):
            # Update monitors
            self._update_monitors()

            # STDP updates according to incoming new pre-spikes
            self._update_on_pre_spikes(self.time)

            # Feed forward
            self._feed_forward()

            # Layers process
            self._process()

            # STDP updates according to incoming new post-spikes
            self._update_on_post_spikes(self.time)

            self.time += 1

            if self.time % 1000 == 0:
                plt.figure(1)
                plt.clf()
                self.plot_weight_map(('I', 'O'), 0.01)
                plt.figure(2)
                plt.clf()
                plt.plot(self.monitors['O'].record['v'].numpy())
                plt.pause(0.01)

    def after_batch(self, keep_count=False):
        """
        Updates network after one batch, e.g. adapts thresholds.
        """
        for lyr in self.layers.values():
            lyr.track_activity()
            lyr.adapt_thresholds()
            if not keep_count:
                lyr.clear_spike_counts()

            lyr.time = 0

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

    def plot_weight_map(self, connection_name, pause_interval):
        """
        Plots weight map of the desired connection.
        :param connection_name:     <tuple>     e.g. ('I', 'O')
        :param pause_interval:      <float>     seconds of pausing time for one plot
        """
        self.connections[connection_name].plot_weight_map(pause_interval)

    def input_image(self, image):
        self.layers['I'].image = image.view(-1)

        # reset membrane potential
        output = self.layers['O']
        output.v = torch.ones_like(output.v) * output.v_rest

    def export_cfg(self):
        """
        Exports a <LocalConfig> object according to current network attributes.
        :return:                <LocalConfig>
        """
        for lyr in self.layers.values():
            lyr.update_cfg()

        for conn in self.connections.values():
            conn.update_cfg()

        self._update_cfg()

        return self.config

    def _update_cfg(self):
        # mutable config
        self.config.input.average_firing_rate = self.input_firing_rate
