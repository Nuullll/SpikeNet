# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: neuron.py
# @Last modified by:   Yilong Guo
# @Last modified time: 04-Sep-2018


import random


class Neuron:
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
        Reset the internal state.
        """
        raise NotImplementedError('Neuron.reset() is not implemented.')


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
        self.state += 1
        if self.state == self.firing_step:
            # fires a spike
            self.output = self.spike_amp
            self.reset()

    def reset(self):
        """
        Reset `self.state` to 0.
        """
        self.state = 0
