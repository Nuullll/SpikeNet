# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: layer.py
# @Last modified by:   Yilong Guo
# @Last modified time: 04-Sep-2018


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
        self.input = torch.zeros_like(state, dtype=torch.float)
        self.output = torch.zeros_like(state, dtype=torch.float)
        self.firing_mask = torch.zeros_like(state, dtype=torch.uint8)

        # global config for <Layer> class
        self.config = {
            'rest': 0.0,        # rest potential of output signal
            'peak': 1.0         # peak potential of output signal when firing a spike
        }

    def process(self):
        """
        Processes the layer by one time step.
        """
        raise NotImplementedError("Layer.process() is not implemented.")

    def _preprocess(self):
        """
        Clears output according to `firing_mask`.
        """
        self.output.masked_fill_(self.firing_mask, self.config['rest'])
        self.firing_mask = torch.zeros_like(self.state, dtype=torch.uint8)

    def _fire(self):
        """
        Fires spikes according to `firing_mask`.
        """
        self.output.masked_fill_(self.firing_mask, self.config['peak'])

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
        super(PoissonLayer, self).__init__(state=firing_step/2)     # half way to spike

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


class LIFLayer(Layer):
    """
    Layer of LIF neurons.
    """
    def __init__(self, rest, threshold, leak_factor, refractory):
        """
        :param rest:            float       Rest potential.
        :param threshold:       float       Threshold for firing.
        :param leak_factor:     float       Leak factor between 0 and 1.
        :param refractory:      int         Refractory period. (unit: time step)
        """
        self.config['rest'] = rest
        self.config['threshold'] = threshold
        self.config['leak_factor'] = leak_factor
        self.config['refractory'] = refractory
