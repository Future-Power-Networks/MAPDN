import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utilities.util import *


class ReinforcementLearning(object):
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.cuda_ = torch.cuda.is_available() and self.args.cuda

    def __str__(self):
        print (self.name)

    def __call__(self):
        raise NotImplementedError()

    def get_loss(self):
        raise NotImplementedError()
