import torch
import torch.nn as nn
import numpy as np
from utilities.util import select_action, cuda_wrapper, batchnorm
from models.model import Model
from learning_algorithms.ppo import PPO
from collections import namedtuple
from critics.mlp_critic import MLPCritic


class MAPPO(Model):
    def __init__(self, args, target_net=None):
        super(MAPPO, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.rl = PPO(self.args)

    def construct_value_net(self):
        if self.args.agent_id:
            input_shape = self.obs_dim * self.n_ + self.n_ # it is a v(s) rather than q(s, a)
        else:
            input_shape = self.obs_dim * self.n_ # it is a v(s) rather than q(s, a)
        output_shape = 1
        self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) ] )

    def construct_policy_net(self):
        if self.args.agent_type == 'mlp':
            from agents.mlp_agent_ppo import MLPAgent
            if self.args.agent_id:
                self.policy_dicts = nn.ModuleList([ MLPAgent(self.obs_dim + self.n_, self.args) ])
            else:
                self.policy_dicts = nn.ModuleList([ MLPAgent(self.obs_dim, self.args) ])
        elif self.args.agent_type == 'rnn':
            from agents.rnn_agent_ppo import RNNAgent
            if self.args.agent_id:
                self.policy_dicts = nn.ModuleList([ RNNAgent(self.obs_dim + self.n_, self.args) ])
            else:
                self.policy_dicts = nn.ModuleList([ RNNAgent(self.obs_dim, self.args) ])
        else:
            NotImplementedError()

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def value(self, obs, act):
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)
        batch_size = obs.size(0)

        # obs_own = obs.clone()
        obs = obs.unsqueeze(1).expand(batch_size, self.n_, self.n_, self.obs_dim) # shape = (b, n, o) -> (b, 1, n, o) -> (b, n, n, o)
        obs = obs.contiguous().view(batch_size, self.n_, -1) # shape = (b, n, o*n)

        # add agent id
        if self.args.agent_id:
            agent_ids = torch.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1) # shape = (b, n, n)
            agent_ids = cuda_wrapper(agent_ids, self.cuda_)
            inp = torch.cat( (obs, agent_ids), dim=-1 ) # shape = (b, n, o*n+n)
        else:
            inp = obs
            
        inputs = inp.contiguous().view(batch_size*self.n_, -1)
        agent_value = self.value_dicts[0]
        values, _ = agent_value(inputs, None)
        values = values.contiguous().view(batch_size, self.n_, -1)

        return values

    def get_actions(self, state, status, exploration, actions_avail, target=False, last_hid=None):
        if self.args.continuous:
            means, log_stds, hiddens = self.policy(state, last_hid=last_hid) if not target else self.target_net.policy(state, last_hid=last_hid)
            if means.size(-1) > 1:
                means_ = means.sum(dim=1, keepdim=True)
                log_stds_ = log_stds.sum(dim=1, keepdim=True)
            else:
                means_ = means
                log_stds_ = log_stds
            actions, log_prob_a = select_action(self.args, means_, status=status, exploration=exploration, info={'log_std': log_stds_})
            restore_mask = 1. - cuda_wrapper((actions_avail == 0).float(), self.cuda_)
            restore_actions = restore_mask * actions
            action_out = (means, log_stds)
        else:
            logits, _, hiddens = self.policy(state, last_hid=last_hid) if not target else self.target_net.policy(state, last_hid=last_hid)
            logits[actions_avail == 0] = -9999999
            actions, log_prob_a = select_action(self.args, logits, status=status, exploration=exploration)
            restore_actions = actions
            action_out = logits
        return actions, restore_actions, log_prob_a, action_out, hiddens

    def get_loss(self, batch):
        policy_loss, value_loss, action_out = self.rl.get_loss(batch, self, self.target_net)
        return policy_loss, value_loss, action_out