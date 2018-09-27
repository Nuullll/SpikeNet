# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: connection.py
# @Last modified by:   Yilong Guo
# @Last modified time: 27-Sep-2018


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

    def feed_forward(self):
        """
        Fetches output of `pre_layer` and computes results as input of `post_layer`.
        """
        pre = self.pre_layer.o
        self.post_layer.i = torch.matmul(pre, self.weight)
