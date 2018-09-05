# @Author: Yilong Guo
# @Date:   04-Sep-2018
# @Email:  vfirst218@gmail.com
# @Filename: monitor.py
# @Last modified by:   Yilong Guo
# @Last modified time: 04-Sep-2018


import torch
from .layer import *


class Monitor:
    """
    Record state variables in specific <Layer>.
    """
    def __init__(self, target, state_vars):
        """
        Creates a <Monitor> recording `state_vars` in `target` layer.
        :param target:          <Layer>
        :param state_vars:      [str]
        """
        self.target = target
        self.state_vars = state_vars
        self.record = {state: torch.tensor([]) for state in state_vars}

    def update(self):
        """
        Gets `state` values from `target` <Layer>.
        Updates `self.record`.
        """
        for state, history in self.record.items():
            self.record[state] = torch.cat((history, self.target.get_state(state)), 0)
