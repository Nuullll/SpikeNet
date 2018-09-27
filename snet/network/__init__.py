# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: __init__.py
# @Last modified by:   Yilong Guo
# @Last modified time: 04-Sep-2018


from .layer import *
from .connection import *
from .monitor import *


class NetworkBuilder:
    """
    Network builder, with given <Layer>s, <torch.Tensor> weights, <Monitor>s.
    """
    def __init__(self, layers, weights, monitors, dt=1.0):
        """
        Initialize builder.
        :param layers:      dict        {name: <Layer>}
        :param weights:     dict        {(pre_layer_name, post_layer_name): weights}
        :param monitors:    dict        {name: [state_vars]}
        :param dt:          float       Simulation time step.
        """
        self.network = Network(dt=dt)

        # initialize layers
        self.network.layers = layers

        # initialize connections
        self._init_connections(weights)

        # initialize monitors
        self._init_monitors(monitors)

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

    def build(self):
        """
        :return:    <Network>
        """
        return self.network


class Network:
    """
    Responsible for interaction simulation among neurons and synapses.
    """
    def __init__(self, dt=1.0):
        """
        Initialize <Network> instance, with a configuration.
        """
        self.layers = {}            # {name: <Layer>}
        self.connections = {}       # {(pre_layer_name, post_layer_name): <Connection>}
        self.monitors = {}          # {layer_name: <Monitor>}

        self.dt = dt

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

            # Fetch pre-spikes
            self._record_pre_spikes(t)

            # Feed forward
            self._feed_forward()

            # Layers process
            self._process()

            # Fetch post-spikes
            self._record_post_spikes(t)

            # Update weights

    def _update_monitors(self):
        """
        Feeds state values to monitors.
        """
        for mon in self.monitors.values():
            mon.update()

    def _record_pre_spikes(self, time):
        """
        Feeds pre-spikes to <Connection>s.
        """
        for conn in self.connections.values():
            conn.record_pre_spikes(time)

    def _feed_forward(self):
        """
        Feeds pre-synaptic signals to synapses, and then pass results to post-synaptic neurons' input.
        """
        for conn in self.connections.values():
            conn.feed_forward()

    def _record_post_spikes(self, time):
        """
        Feeds post-spikes to <Connection>s.
        """
        for conn in self.connections.values():
            conn.record_post_spikes(time)

    def _process(self):
        """
        Each layer processes. Generates new output for each layer.
        """
        for lyr in self.layers.values():
            lyr.process()

    def _update_weights(self):
        """
        Updates weights, according to rules config of <Connection>.
        """
        pass
