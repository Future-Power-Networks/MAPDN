from .voltage_loss_registry import Voltage_loss



class VoltageLoss(object):
    def __init__(self, name):
        self.name = name
        self.reward = Voltage_loss[name]

    def step(self, vs):
        return self.reward(vs)
