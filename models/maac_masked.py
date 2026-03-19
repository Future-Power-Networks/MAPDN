import torch.nn as nn

from critics.masked_maac_critic import MaskedAttentionCritic
from models.maac import MAAC


class MaskedMAAC(MAAC):
    """
    Keep MAPDN's original MAAC actor and training logic, but swap the critic.
    """

    def construct_value_net(self):
        self.value_dicts = nn.ModuleList([MaskedAttentionCritic(self.args)])
