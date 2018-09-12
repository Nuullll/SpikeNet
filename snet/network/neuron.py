# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: neuron.py
# @Last modified by:   Yilong Guo
# @Last modified time: 04-Sep-2018


import random


class Neuron(object):
    """
    Abstract base class for a neuron.
    """
    def __init__(self, state, **kwargs):
        """
        Initialize <Neuron>.
        :param state:       *       Depends on concrete neuron type.
        """
        self.state = state
        self.input = 0.0
        self.output = 0.0
        self.clear_output = False

        # optional config
        self.spike_amp = kwargs.get('spike_amp', 1.0)    # The default output amplitude

    def process(self):
        """
        Processes `input` signal according to `state` itself, and gives `output`.
        Processes only 1 time step at each call.
        Updates `self.state` and `self.output`.
        """
        raise NotImplementedError('Neuron.process() is not implemented.')

    def reset(self):
        """
        Resets the internal state.
        """
        raise NotImplementedError('Neuron.reset() is not implemented.')

    def fire_and_reset(self):
        """
        Fires a spike and resets the neuron.
        """
        self.output = self.spike_amp
        self.clear_output = True
        self.reset()

    def preprocess(self):
        """
        Clears output after firing.
        """
        if self.clear_output:
            self.output = 0.0
            self.clear_output = False


class PoissonNeuron(Neuron):
    """
    Outputs Poisson spike trains, often used as Input layer.
    """
    def __init__(self, firing_step, **kwargs):
        """
        A Poisson neuron fires every `firing_step` steps.
        `self.state` serves as a counter to emit spikes.
        :param firing_step:         int         Unit: time step.
        """
        self.firing_step = firing_step
        super(PoissonNeuron, self).__init__(state=random.randint(0, firing_step-1), **kwargs)

    def process(self):
        """
        Generates Poisson spike trains. Firing rate equals to `state`.
        Outputs a spike after every `firing_step` calls.
        """
        self.preprocess()

        self.state += 1
        if self.state == self.firing_step:
            # fire and reset
            self.fire_and_reset()

    def reset(self):
        """
        Reset `self.state` to 0.
        """
        self.state = 0


class LIFNeuron(Neuron):
    """
    Leaky-Integrate-and-Fire neuron model.
    """
    def __init__(self, rest, threshold, leak_factor, **kwargs):
        """
        Initialize a LIF Neuron.
        :param rest:            float       Rest potential.
        :param threshold:       float       Spiking threshold.
        :param leak_factor:     float       Leak factor, less than 1 (indicates leaking)
        :param kwargs:          dict        Optional parameters.
        """
        self.rest = rest
        self.threshold = threshold
        self.leak_factor = leak_factor

        # The membrane potential serves as the internal state.
        # Initialize to rest potential.
        super(LIFNeuron, self).__init__(state=rest, **kwargs)

    def process(self):
        """
        Leaks, integrates, and fires.
        Updates `self.state` and `self.output`.
        """
        self.preprocess()

        # leak
        self.v -= self.leak_factor * (self.v - self.rest)

        # integrate
        self.v += self.input

        # fire
        if self.v >= self.threshold:
            self.fire_and_reset()

    def reset(self):
        """
        Reset membrane potential to rest potential.
        """
        self.state = self.rest

    @property
    def v(self):
        """
        Membrane potential property, returns `self.state`.
        For convenience.
        """
        return self.state

    @v.setter
    def v(self, value):
        self.state = value
