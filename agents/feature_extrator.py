import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class FeatureExtractor(nn.Module):
    def __init__(self, input_shape, args):
        super(FeatureExtractor, self).__init__()
        self.args = args

        # Easiest to reuse hid_size variable
        self.cnn1d = nn.Conv1d(in_channels=args.obs_hist_len,
                               out_channels=1, 
                               kernel_size=3)
        self.output_shape = np.floor(input_shape-2)

    def forward(self, inputs):
        x = self.cnn1d(inputs)
        f = th.tanh(x)
        return f