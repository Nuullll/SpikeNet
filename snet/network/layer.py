# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: layer.py
# @Last modified by:   Yilong Guo
# @Last modified time: 27-Sep-2018


import torch


class Layer(object):
    """
    Abstract class <Layer>.
    """

    def __init__(self, state, config):
        """
        Each neuron serves as a FSM.
        :param state:       <torch.Tensor>      Inner state tensor for all neurons.
        """
        # state variables
        self.size = state.shape[0]
        self.state = state
        self.i = torch.zeros_like(state, dtype=torch.float)         # input port
        self.o = torch.zeros_like(state, dtype=torch.float)         # output port
        self.firing_mask = torch.zeros_like(state, dtype=torch.uint8)
        # spike counter
        self.spike_counts = torch.zeros_like(state, dtype=torch.long)

        # adaptive thresholds
        self.adaptive = False

        # lateral inhibition
        self.inhibition = False

        # config
        self.config = config
        # for performance
        self.o_rest = self.config.layer_basics.o_rest
        self.o_peak = self.config.layer_basics.o_peak

    def process(self):
        """
        Processes the layer by one time step.
        """
        raise NotImplementedError("Layer.process() is not implemented.")

    def _preprocess(self):
        """
        Clears output according to `firing_mask`.
        """
        self.o.masked_fill_(self.firing_mask, self.o_rest)
        self.firing_mask = torch.zeros_like(self.state, dtype=torch.uint8)

    def _fire(self):
        """
        Fires spikes according to `firing_mask`.
        """
        self.o.masked_fill_(self.firing_mask, self.o_peak)
        self.spike_counts += torch.where(self.firing_mask, torch.ones_like(self.spike_counts),
                                         torch.zeros_like(self.spike_counts))

    def _reset(self):
        """
        Resets internal state.
        """
        raise NotImplementedError("Layer._reset() is not implemented.")

    def _fire_and_reset(self):
        """
        Invokes `_fire()` and `_reset()`.
        """
        self._fire()
        self._reset()

    def adapt_thresholds(self):
        pass

    def clear_spike_counts(self):
        """
        Clears self.spike_counts.
        """
        self.spike_counts = torch.zeros_like(self.spike_counts, dtype=torch.long)

    def update_cfg(self):
        self.config.layer_basics.o_rest = self.o_rest
        self.config.layer_basics.o_peak = self.o_peak


class PoissonLayer(Layer):
    """
    Layer of Poisson neurons.
    """
    def __init__(self, firing_step, config):
        """
        :param firing_step:         <torch.IntTensor>       Specify firing step of each neuron.
        """
        self.firing_step = firing_step

        state = firing_step.float() * torch.rand_like(firing_step.float())
        super(PoissonLayer, self).__init__(state=state.long(), config=config)

    def process(self):
        """
        Generates a spike every `firing_step` calls.
        """
        self._preprocess()

        self.state += 1
        # fire spikes
        self.firing_mask = (self.state >= self.firing_step)
        self._fire_and_reset()

    def _reset(self):
        """
        Resets `state` to `0`.
        """
        self.state.masked_fill_(self.firing_mask, 0)

    def set_step(self, firing_step):
        """
        Sets new firing steps. Resets `self.state`.
        :param firing_step:     <torch.IntTensor>
        """
        self.firing_step = firing_step
        self.state = (firing_step.float() * torch.rand_like(firing_step.float())).long()

    def update_cfg(self):
        super(PoissonLayer, self).update_cfg()


class LIFLayer(Layer):
    """
    Layer of LIF neurons.
    """
    def __init__(self, size, config):
        """
        :param size:            int         Number of neurons in this layer.
        """
        # get config
        self.config = config
        # for performance
        self.v_rest = self.config.lif_layer.v_rest
        self.v_th_rest = self.config.lif_layer.v_th_rest
        self.dv_th = self.config.lif_layer.dv_th
        self.refractory = self.config.lif_layer.refractory
        self.tau = self.config.lif_layer.tau
        self.res = self.config.lif_layer.res
        self.winners = self.config.lif_layer.winners

        # the membrane potential serves as the internal state
        super(LIFLayer, self).__init__(state=torch.ones(size) * self.v_rest, config=config)

        self.size = size

        self.v_th = torch.ones(size, dtype=torch.float) * self.v_th_rest

        # record the steps from last spike timing to now
        self._spike_history = torch.ones(size, dtype=torch.int) * self.refractory

        # adaptive threshold
        self.adaptive = True

        # lateral inhibition
        self.inhibition = True

    @property
    def v(self):
        """
        Alias of `self.state`, i.e. the membrane potential.
        """
        return self.state

    @v.setter
    def v(self, value):
        """
        Sets `self.state`.
        """
        self.state = value

    def process(self):
        """
        Leaks, integrates and fires.
        """
        self._preprocess()

        # leak
        self.v -= (self.v - self.v_rest) / self.tau

        # during refractory period?
        self._spike_history += 1
        active = (self._spike_history >= self.refractory)

        # integrate (on active neurons)
        self.v += torch.where(active, self.res / self.tau * self.i, torch.zeros_like(self.i))   # coef = 1/C

        # lateral inhibition
        if self.inhibition:
            # candidates = (self.v >= self.v_th).nonzero()
            #
            # if len(candidates) > 0:
            #     inds = (self.v >= self.v_th).nonzero().squeeze(1)
            #
            #     perm = torch.randperm(inds.size(0))
            #
            #     for p in perm:
            #         ind = inds[p]
            #         overshoot = self.v[ind] - self.v_th[ind]
            #         if overshoot < 0:
            #             continue
            #
            #         mask = torch.ones_like(self.firing_mask)
            #         mask.scatter_(0, ind, 0)
            #         # self.v.masked_scatter_(mask, self.v - self.dv_inh).clamp_(min=self.v_rest)
            #         self.v.masked_fill_(mask, self.v_rest)

            overshoot = self.v - self.v_th
            overshoot_mask = overshoot > 0
            _, indices = torch.sort(overshoot, descending=True)

            overshoot_mask = overshoot_mask.index_select(0, indices)

            indices = indices.masked_select(overshoot_mask)

            indices = indices[:self.winners]
            # indices = indices[:round(self.winners * overshoot_mask.sum().item() / self.size)]

            if len(indices) > 0:
                mask = torch.ones_like(self.firing_mask)
                mask.scatter_(0, indices, 0)
                self.v.masked_fill_(mask, self.v_rest)

        # ready to fire
        self.firing_mask = (self.v >= self.v_th)

        # if self.adaptive:
        #     # thresholds integrate (on firing neurons)
        #     # self.v_th = torch.where(self.firing_mask, self.v_th + self.dv_th, self.v_th)
        #     self.v_th = torch.where(self.firing_mask, self.v, self.v_th)

        # fire and reset
        self._fire_and_reset()

    def _reset(self):
        """
        Resets fired neurons' potentials to `self.v_rest`.
        Starts refractory process.
        """
        self.v.masked_fill_(self.firing_mask, self.v_rest)
        self._spike_history.masked_fill_(self.firing_mask, 0)

    def adapt_thresholds(self):
        """
        Adapts thresholds.
        """
        if self.adaptive:
            # increase the threshold of the most recent active neuron, according to self.spike_counts
            _, indices = torch.sort(self.spike_counts, descending=True)

            idx = indices[:self.winners]

            mask = torch.zeros_like(self.firing_mask)
            mask.scatter_(0, idx, 1)

            d = -self.dv_th * torch.ones_like(self.v) / (self.size - self.winners) * self.winners
            d.masked_fill_(mask, self.dv_th)

            self.v_th += d
            # self.v_th.clamp_(min=self.v_th_rest)

    def update_cfg(self):
        super(LIFLayer, self).update_cfg()

        self.config.lif_layer.v_rest = self.v_rest
        self.config.lif_layer.v_th_rest = self.v_th_rest
        self.config.lif_layer.dv_th = self.dv_th
        self.config.lif_layer.refractory = self.refractory
        self.config.lif_layer.tau = self.tau
        self.config.lif_layer.res = self.res
        self.config.lif_layer.winners = self.winners
