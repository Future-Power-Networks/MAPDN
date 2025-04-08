import torch as th
import torch.nn as nn



class MLPAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MLPAgent, self).__init__()
        self.args = args

        # Easiest to reuse hid_size variable
        self.fc1 = nn.Linear(input_shape, args.hid_size)

        if args.layernorm:
            self.layernorm = nn.LayerNorm(args.hid_size)
            
        self.fc2 = nn.Linear(args.hid_size, args.hid_size)
        self.mean = nn.Linear(args.hid_size, args.action_dim)
        self.log_std = nn.Linear(args.hid_size, args.action_dim)

        if self.args.hid_activation == 'relu':
            self.hid_activation = nn.ReLU()
        elif self.args.hid_activation == 'tanh':
            self.hid_activation = nn.Tanh()
        
    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.agent_num, self.args.hid_size).zero_()

    def forward(self, inputs, hidden_state):
        x = self.fc1(inputs)
        if self.args.layernorm:
            x = self.layernorm(x)
        x = self.hid_activation(x)
        h = self.hid_activation(self.fc2(x))
        mean = self.mean(h)
        log_std = self.log_std(h)
        log_std = th.tanh(log_std)
        log_std = self.args.LOG_STD_MIN + 0.5 * (self.args.LOG_STD_MAX - self.args.LOG_STD_MIN) * (log_std + 1) # From SpinUp / Denis Yarats
        return mean, log_std, h