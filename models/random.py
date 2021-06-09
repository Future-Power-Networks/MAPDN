import torch
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from learning_algorithms.ddpg import *
from collections import namedtuple



class RandomAgent(Model):

    def __init__(self, args):
        super(RandomAgent, self).__init__(args)
        self.args = args

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        actions = []
        tensor = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
        actions = tensor([[1.0]*self.act_dim]*self.n_)
        return actions
