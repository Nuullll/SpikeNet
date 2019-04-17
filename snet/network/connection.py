# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: connection.py
# @Last modified by:   Yilong Guo
# @Last modified time: 27-Sep-2018


from .layer import *
import matplotlib.pyplot as plt
import math
from snet.device import RRAM_device_variation


class Connection:
    """
    <Connection> describes the weight matrix between two <Layer>s of <Neuron>s.
    """
    def __init__(self, pre_layer, post_layer, weight, config):
        """
        Initialize a <Connection> which connects from `pre_layer` to `post_layer`.
        :param pre_layer:       <Layer>
        :param post_layer:      <Layer>
        :param weight:          <torch.Tensor>      Matches the sizes of `pre_layer` and `post_layer`.
        """
        self.pre_layer = pre_layer
        self.post_layer = post_layer
        self.weight = weight

        self._last_pre_spike_time = -torch.ones(self.pre_layer.size)        # -1 means never fired
        self._last_post_spike_time = -torch.ones(self.post_layer.size)

        # static mode (weight will not change)
        self.static = False

        # get config
        self.config = config
        # for performance
        self.decay = self.config.synapse.decay_eff
        self.learn_rate_p = self.config.synapse.learn_rate_p_eff
        self.learn_rate_m = self.config.synapse.learn_rate_m_eff
        self.tau_p = self.config.synapse.tau_p
        self.tau_m = self.config.synapse.tau_m
        self.w_min = self.config.synapse.w_min
        self.w_max = self.config.synapse.w_max
        self.variation = self.config.synapse.variation

    def feed_forward(self):
        """
        Fetches output of `pre_layer` and computes results as input of `post_layer`.
        """
        pre = self.pre_layer.o
        self.post_layer.i = torch.matmul(pre, self.weight)

    def _clamp(self):
        self.weight.clamp_(min=self.w_min, max=self.w_max)

    def update_on_pre_spikes(self, time):
        """
        Updates weights when new pre-spikes come.
        """
        if self.static:
            return

        active = torch.ger(self.pre_layer.firing_mask, self.post_layer.last_firing_mask)

        self.weight[active] -= self.learn_rate_m * (self.weight[active] - self.w_min)
        self._clamp()

    def update_on_post_spikes(self, time):
        """
        Updates weights when new post-spikes come.
        """
        if self.static:
            return

        # pre-spikes and post-spikes at the same time
        active = torch.ger(self.pre_layer.firing_mask, self.post_layer.firing_mask)

        self.weight[active] += self.learn_rate_p * (self.w_max - self.weight[active])
        self._clamp()

    def plot_weight_map(self, pause_interval):
        """
        Plots weight map.
        """
        # get config
        output_num = self.config.network.output_neuron_number
        col_num = math.ceil(math.sqrt(output_num))
        row_num = math.ceil(output_num / col_num)
        width = self.config.input.image_width
        height = self.config.input.image_height
        w_min = self.config.synapse.w_min
        w_max = self.config.synapse.w_max

        # plot
        # plt.figure()
        for i in range(output_num):
            plt.subplot(row_num, col_num, i + 1)
            plt.matshow(self.weight[:, i].view(width, height), fignum=False, vmin=w_min, vmax=w_max)

        plt.pause(pause_interval)

    def update_cfg(self):
        self.config.synapse.decay_eff = self.decay
        self.config.synapse.learn_rate_p_eff = self.learn_rate_p
        self.config.synapse.learn_rate_m_eff = self.learn_rate_m
        self.config.synapse.tau_p = self.tau_p
        self.config.synapse.tau_m = self.tau_m
        self.config.synapse.w_min = self.w_min
        self.config.synapse.w_max = self.w_max
        self.config.synapse.variation = self.variation
