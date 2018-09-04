# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: neuron.py
# @Last modified by:   Yilong Guo
# @Last modified time: 04-Sep-2018


class Neuron:
    """
    Abstract base class for a neuron.
    """
    def __init__(self, **kwargs):
        """
        Initialize <Neuron>.
        :param kwargs:      dict        Contains 'state', 'delay' options.
        """
        self.state = kwargs.get('state', None)
        self.delay = kwargs.get('delay', 1)         # Processing delay, default: 1 time step
        self.input = None
        self.output = None

    def process(self):
        """
        Processes `input` signal according to `state` itself, and gives `output`,
        which finishes in `delay` time steps.
        :return:        None        Updates `self.state` and `self.output`.
        """
        raise NotImplementedError('Neuron.process() is not implemented.')

    