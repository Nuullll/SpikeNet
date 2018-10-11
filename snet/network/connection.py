# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: connection.py
# @Last modified by:   Yilong Guo
# @Last modified time: 27-Sep-2018


import math
from .layer import *


class Connection:
    """
    <Connection> describes the weight matrix between two <Layer>s of <Neuron>s.
    """
    def __init__(self, pre_layer, post_layer, weight):
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

        # STDP parameters
        self.learn_rate = 0.5
        self.w_min = 1e-5
        self.w_max = 1e-4

    def feed_forward(self):
        """
        Fetches output of `pre_layer` and computes results as input of `post_layer`.
        """
        pre = self.pre_layer.o
        self.post_layer.i = torch.matmul(pre, self.weight)

    def update_on_pre_spikes(self, time):
        """
        Updates weights when new pre-spikes come.
        """
        # record new pre-spikes
        self._last_pre_spike_time.masked_fill_(self.pre_layer.firing_mask, time)

        # mask
        post_active = self._last_post_spike_time >= 0
        active = torch.ger(self.pre_layer.firing_mask, post_active)     # new pre-spikes and fired post-spikes

        # calculate timing difference (where new pre-spikes timing is now)
        dt = self._last_pre_spike_time.repeat(self.post_layer.size, 1).t() - \
            self._last_post_spike_time.repeat(self.pre_layer.size, 1)

        # weights decrease, because pre-spikes come after post-spikes
        dw = self.learn_rate * (self.weight - self.w_min) * torch.exp(-dt)
        dw.masked_fill_(~active, 0)
        self.weight -= dw

    def update_on_post_spikes(self, time):
        """
        Updates weights when new post-spikes come.
        """
        # record new post-spikes
        self._last_post_spike_time.masked_fill_(self.post_layer.firing_mask, time)

        # mask
        pre_active = self._last_pre_spike_time >= 0
        active = torch.ger(pre_active, self.post_layer.firing_mask)     # new post-spikes and fired pre-spikes

        # calculate timing difference (where new post-spikes timing is now)
        dt = self._last_post_spike_time.repeat(self.pre_layer.size, 1) - \
            self._last_pre_spike_time.repeat(self.post_layer.size, 1).t()

        # weights increase, because post-spikes come after pre-spikes
        dw = self.learn_rate * (self.w_max - self.weight) * torch.exp(-dt)
        dw.masked_fill_(~active, 0)
        self.weight += dw
