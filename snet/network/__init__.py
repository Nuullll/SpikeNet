# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: __init__.py
# @Last modified by:   Yilong Guo
# @Last modified time: 04-Sep-2018


from .layer import *


class NetworkConfig:
    """
    Configuration for <Network>.
    """
    def __init__(self):
        self.dt = 1.0
        self.layers = {}
        self.connections = {}
        self.monitors = {}


class Network:
    """
    Responsible for interaction simulation among neurons and synapses.
    """
    def __init__(self, config):
        """
        Initialize <Network> instance, with a configuration.
        :param config:      <NetworkConfig>         Configuration.
        """
        self.config = config
        self.layers = config.layers
        self.connections = config.connections
        self.monitors = config.monitors
