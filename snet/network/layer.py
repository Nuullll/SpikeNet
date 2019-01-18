# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: layer.py
# @Last modified by:   Yilong Guo
# @Last modified time: 27-Sep-2018


import torch
import random


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

        # activity tracker
        self.activity_history = torch.tensor([])

        # adaptive thresholds
        self.adaptive = False

        # lateral inhibition
        self.inhibition = False

        # config
        self.config = config
        # for performance
        self.o_rest = self.config.layer_basics.o_rest
        self.o_peak = self.config.layer_basics.o_peak

        self.track_phase = self.config.lif_layer.track_phase

        self.time = 0

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

    def track_activity(self):
        """
        Track neurons' recent activity.
        """
        self.activity_history = torch.cat((self.activity_history, self.spike_counts.float().unsqueeze(0)), 0)

        if len(self.activity_history) > self.track_phase:
            self.activity_history = self.activity_history[-self.track_phase:]

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
    def __init__(self, config):
        super(PoissonLayer, self).__init__(state=torch.zeros(config.network.input_neuron_number), config=config)
        self.image = None

        self.pattern_firing_rate = self.config.input.pattern_firing_rate
        self.background_firing_rate = self.config.input.background_firing_rate
        self.dt = self.config.network.dt_s

    def process(self):
        self._preprocess()

        x = torch.rand_like(self.image, dtype=torch.float)
        ref = self.pattern_firing_rate * self.image.float() / self.image.float().sum()
        # ref.masked_fill_(self.image == 0, self.background_firing_rate * self.dt)

        # fire spikes
        self.firing_mask = (x <= ref)
        self._fire_and_reset()

    def _reset(self):
        """
        Resets `state` to `0`.
        """
        self.state.masked_fill_(self.firing_mask, 0)

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
        self.inputs = self.config.network.input_neuron_number

        self.firing_event_target = self.config.lif_layer.firing_event_target
        self.activated_phase_target = self.config.lif_layer.activated_phase_target
        self.duration_per_training_image = self.config.input.duration_per_training_image

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
        self.v += torch.where(active, self.res * self.i / self.inputs, torch.zeros_like(self.i))   # coef = 1/C

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
        #     th_decay = 0.00005
        #     self.v_th = torch.where(self.firing_mask, self.v_th + self.dv_th,
        #                             self.v_th - th_decay * (self.v_th - self.v_th_rest))

        # fire and reset
        self._fire_and_reset()

        self.time += 1

    def _reset(self):
        """
        Resets fired neurons' potentials to `self.v_rest`.
        Starts refractory process.
        """
        # # adapt thresholds after firing
        if self.adaptive:
            self.v_th += self.dv_th * self.firing_mask.float()

        self.v.masked_fill_(self.firing_mask, self.v_rest)
        self._spike_history.masked_fill_(self.firing_mask, 0)

    def adapt_thresholds(self):
        """
        Adapts thresholds.
        """
        if self.adaptive:
            if len(self.activity_history) > 0:
                last_activity = self.activity_history[-1].float()

                total_firing_events = self.activity_history.sum(0).float()

                factor = total_firing_events / len(self.activity_history) / (
                        self.firing_event_target / self.size)
                factor[torch.isnan(factor)] = 1
                factor.clamp_(min=0.5)

                # slow down
                factor = torch.ones_like(factor) + (factor - torch.ones_like(factor)) / 100

                # if last_activity.sum() > 0:
                #     factor.masked_fill_(last_activity == 0, 1)
                #     factor *= torch.where(last_activity > 0, last_activity / self.firing_event_target,
                #                           torch.ones_like(last_activity))

                self.v_th *= factor

                self.v_th.clamp_(min=2.0, max=20.0)

    def update_cfg(self):
        super(LIFLayer, self).update_cfg()

        self.config.lif_layer.v_rest = self.v_rest
        self.config.lif_layer.v_th_rest = self.v_th_rest
        self.config.lif_layer.dv_th = self.dv_th
        self.config.lif_layer.refractory = self.refractory
        self.config.lif_layer.tau = self.tau
        self.config.lif_layer.res = self.res
        self.config.lif_layer.winners = self.winners
