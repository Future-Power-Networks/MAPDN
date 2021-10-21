import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    # sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1
    def __init__(self, args):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        self.hidden_dim = args.hid_size
        self.attend_heads = args.attend_heads
        assert (self.hidden_dim % self.attend_heads) == 0
        self.sa_sizes = [(args.obs_size, args.action_dim)] * args.agent_num
        self.nagents = args.agent_num
        self.continuous = args.continuous
        self.attend_heads = args.attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.biases = nn.ModuleList()
        self.state_encoders = nn.ModuleList()
        # iterate over agents
        for sdim, adim in self.sa_sizes:
            idim = sdim + adim 
            if args.continuous:
                odim = 1
            else:
                odim = adim
            # s+a encoder
            encoder = nn.Sequential()
            if args.norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim,
                                                            affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, self.hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            # critic
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(2 * self.hidden_dim,
                                                      self.hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(self.hidden_dim, odim))
            self.critics.append(critic)
            # bias
            bias = nn.Sequential()
            bias.add_module('bias_fc1', nn.Linear(self.hidden_dim,
                                                  self.hidden_dim))
            bias.add_module('bias_nl', nn.LeakyReLU())
            bias.add_module('bias_fc2', nn.Linear(self.hidden_dim, 1))
            self.biases.append(bias)
            # s encoder
            state_encoder = nn.Sequential()
            if args.norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(
                                            sdim, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(sdim,
                                                            self.hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = self.hidden_dim // self.attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for _ in range(self.attend_heads):
            self.key_extractors.append(nn.Linear(self.hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(self.hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(self.hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

    def forward(self, inps, return_q=True, regularize=True):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        agents = range(len(self.critic_encoders))
        states, actions, sa = inps
        # extract state-action encoding for each agent
        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, sa)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                # calculate attention across agents
                attend_logits = th.matmul(selector.view(selector.shape[0], 1, -1),
                                             th.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (th.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        all_rets = []
        for i, a_i in enumerate(agents):
            agent_rets = []
            if self.continuous:
                critic_in = th.cat((sa_encodings[i], *other_all_values[i]), dim=1)
                all_q = self.critics[a_i](critic_in)
                q = all_q
            else:
                critic_in = th.cat((s_encodings[i], *other_all_values[i]), dim=1)
                all_q = self.critics[a_i](critic_in)
                int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
                q = all_q.gather(1, int_acs)
            bias_in = s_encodings[i]
            b = self.biases[a_i](bias_in)
            if return_q:
                agent_rets.append(q - b)
            if regularize:
                # regularize magnitude of attention logits
                attend_mag_reg = 1e-3 * sum((logit**2).mean() for logit in
                                            all_attend_logits[i])
                regs = attend_mag_reg.view(1,1)
                agent_rets.append(regs)
            all_rets.append(agent_rets)
        return all_rets