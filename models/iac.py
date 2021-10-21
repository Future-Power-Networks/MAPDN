import torch as th
import torch.nn as nn
import numpy as np
from utilities.util import *
from models.model import Model
from collections import namedtuple
from learning_algorithms.actor_critic import ActorCritic
from critics.mlp_critic import MLPCritic


class IAC(Model):
    def __init__(self, args, target_net=None):
        super(IAC, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.rl = ActorCritic(self.args)

    def construct_value_net(self):
        if self.args.continuous:
            if self.args.agent_id:
                input_shape = self.obs_dim + self.act_dim + self.n_
            else:
                input_shape = self.obs_dim + self.act_dim
            output_shape = 1
            if self.args.shared_params:
                self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) ] )
            else:
                self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) for _ in range(self.n_) ] )
        else:
            if self.args.agent_id:
                input_shape = self.obs_dim + self.n_
            else:
                input_shape = self.obs_dim
            output_shape = self.act_dim
            if self.args.shared_params:
                self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) ] )
            else:
                self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) for _ in range(self.n_) ] )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def value(self, obs, act=None):
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)
        batch_size = obs.size(0)

        # add agent id
        if self.args.agent_id:
            agent_ids = th.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device) # shape = (b, n, n)
            obs = th.cat( (obs, agent_ids), dim=-1 ) # shape = (b, n, o+n)

        if self.args.shared_params:
            obs = obs.contiguous().view(batch_size*self.n_, -1) # shape = (b*n, o+n/o)
            act = act.contiguous().view(batch_size*self.n_, -1) # shape = (b*n, a)

        if self.args.continuous:
            inputs = th.cat([obs, act], dim=-1)
        else:
            inputs = obs

        if self.args.shared_params:
            agent_value = self.value_dicts[0]
            values, _ = agent_value(inputs, None)
            values = values.contiguous().view(batch_size, self.n_, -1)
        else:
            values = []
            for i, agent_value in enumerate(self.value_dicts):
                value, _ = agent_value(inputs[:, i, :], None)
                values.append(value)
            values = th.stack(values, dim=1)

        return values

    def get_actions(self, state, status, exploration, actions_avail, target=False, last_hid=None):
        target_policy = self.target_net.policy if self.args.target else self.policy
        if self.args.continuous:
            means, log_stds, hid = self.policy(state, last_hid=last_hid) if not target else target_policy(state, last_hid=last_hid)
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
            logits, _, hid = self.policy(state) if not target else target_policy(state, last_hid=last_hid)
            logits[actions_avail == 0] = -9999999
            actions, log_prob_a = select_action(self.args, logits, status=status, exploration=exploration)
            restore_actions = actions
            action_out = logits
        return actions, restore_actions, log_prob_a, action_out, hid

    def get_loss(self, batch):
        policy_loss, value_loss, action_out = self.rl.get_loss(batch, self, self.target_net)
        return policy_loss, value_loss, action_out
