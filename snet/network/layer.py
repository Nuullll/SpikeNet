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

    def __init__(self, state):
        """
        Each neuron serves as a FSM.
        :param state:       <torch.Tensor>      Inner state tensor for all neurons.
        """
        self.size = state.shape[0]
        self.state = state
        self.i = torch.zeros_like(state, dtype=torch.float)         # input port
        self.o = torch.zeros_like(state, dtype=torch.float)         # output port
        self.firing_mask = torch.zeros_like(state, dtype=torch.uint8)

        # global config for <Layer> class
        self.o_rest = 0.0       # rest potential of output signal
        self.o_peak = 1.0       # peak potential of output signal when firing a spike

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


class PoissonLayer(Layer):
    """
    Layer of Poisson neurons.
    """
    def __init__(self, firing_step):
        """
        :param firing_step:         <torch.IntTensor>       Specify firing step of each neuron.
        """
        self.firing_step = firing_step

        state = firing_step.float() * torch.rand_like(firing_step.float())
        super(PoissonLayer, self).__init__(state=state.long())

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


class LIFLayer(Layer):
    """
    Layer of LIF neurons.
    """
    def __init__(self, size, v_rest, v_threshold, leak_factor, refractory):
        """
        :param size:            int         Number of neurons in this layer.
        :param v_rest:          float       Rest potential.
        :param v_threshold:     float       Threshold for firing.
        :param leak_factor:     float       Leak factor between 0 and 1.
        :param refractory:      int         Refractory period. (unit: time step)
        """
        self.size = size
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.leak_factor = leak_factor  # tau = 1/leak_factor = R * C
        self.refractory = refractory
        self.res = 40e6                 # R = 40MOhm

        # record the steps from last spike timing to now
        self._spike_history = torch.ones(size, dtype=torch.int) * refractory

        # the membrane potential serves as the internal state
        super(LIFLayer, self).__init__(state=torch.ones(size) * v_rest)

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
        self.v -= self.leak_factor * (self.v - self.v_rest)

        # during refractory period?
        self._spike_history += 1
        active = (self._spike_history >= self.refractory)

        # integrate (on active neurons)
        self.v += torch.where(active, self.leak_factor * self.res * self.i, torch.zeros_like(self.i))   # coef = 1/C

        # lateral inhibition
        dv = 5

        inds = (self.v - self.v_threshold).nonzero().squeeze()

        perm = torch.randperm(inds.size(0))

        for p in perm:
            ind = inds[p]
            overshoot = self.v[ind] - self.v_threshold
            if overshoot < 0:
                continue

            mask = torch.ones_like(self.firing_mask)
            mask.scatter_(0, ind, 0)
            self.v.masked_scatter_(mask, self.v - dv).clamp_(min=self.v_rest)

        # fire
        self.firing_mask = (self.v >= self.v_threshold)
        self._fire_and_reset()

    def _reset(self):
        """
        Resets fired neurons' potentials to `self.v_rest`.
        Starts refractory process.
        """
        self.v.masked_fill_(self.firing_mask, self.v_rest)
        self._spike_history.masked_fill_(self.firing_mask, 0)
