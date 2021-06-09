import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MLPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args

        # Easiest to reuse hid_size variable
        self.fc1 = nn.Linear(input_shape, args.hid_size)
        self.fc2 = nn.Linear(args.hid_size, args.hid_size)
        self.fc3 = nn.Linear(args.hid_size, args.action_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hid_size).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(x))
        a = self.fc3(h)
        return a, h