import torch as th
import numpy as np
from models.model import Model



class RandomAgent(Model):

    def __init__(self, args):
        super(RandomAgent, self).__init__(args)
        self.args = args

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        # obs_shape = (b, n, o)
        batch_size = obs.size(0)
        means = th.randn(batch_size, self.n_, self.act_dim).to(self.device)
        log_stds = th.zeros_like(means).to(self.device)
        return means, log_stds, None
