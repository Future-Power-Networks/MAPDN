import torch
import torch.nn as nn
import numpy as np
from utilities.util import cuda_wrapper, select_action, batchnorm, prep_obs
from models.model import Model
from collections import namedtuple
from critics.mlp_critic import MLPCritic


class SQDDPG(Model):
    def __init__(self, args, target_net=None):
        super(SQDDPG, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.sample_size = self.args.sample_size

    def construct_value_net(self):
        input_shape = (self.obs_dim + self.act_dim) * self.n_ + self.n_
        # input_shape = (self.obs_dim + self.act_dim) * self.n_
        output_shape = 1
        self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) ] )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def sample_grandcoalitions(self, batch_size):
        seq_set = cuda_wrapper(torch.tril(torch.ones(self.n_, self.n_), diagonal=0, out=None), self.cuda_)
        grand_coalitions = cuda_wrapper(torch.multinomial(torch.ones(batch_size*self.sample_size, self.n_)/self.n_, self.n_, replacement=False), self.cuda_)
        individual_map = cuda_wrapper(torch.zeros(batch_size*self.sample_size*self.n_, self.n_), self.cuda_)
        individual_map.scatter_(1, grand_coalitions.contiguous().view(-1, 1), 1)
        individual_map = individual_map.contiguous().view(batch_size, self.sample_size, self.n_, self.n_)
        subcoalition_map = torch.matmul(individual_map, seq_set)
        grand_coalitions = grand_coalitions.unsqueeze(1).expand(batch_size*self.sample_size, self.n_, self.n_).contiguous().view(batch_size, self.sample_size, self.n_, self.n_) # shape = (b, n_s, n, n)
        return subcoalition_map, grand_coalitions

    def marginal_contribution(self, obs, act):
        batch_size = obs.size(0)
        subcoalition_map, grand_coalitions = self.sample_grandcoalitions(batch_size) # shape = (b, n_s, n, n)
        grand_coalitions = grand_coalitions.unsqueeze(-1).expand(batch_size, self.sample_size, self.n_, self.n_, self.act_dim) # shape = (b, n_s, n, n, a)
        act = act.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_, self.n_, self.act_dim).gather(3, grand_coalitions) # shape = (b, n, a) -> (b, 1, 1, n, a) -> (b, n_s, n, n, a)
        act_map = subcoalition_map.unsqueeze(-1).float() # shape = (b, n_s, n, n, 1)
        act = act * act_map
        act = act.contiguous().view(batch_size, self.sample_size, self.n_, -1) # shape = (b, n_s, n, n*a)
        obs = obs.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_, self.n_, self.obs_dim) # shape = (b, n, o) -> (b, 1, n, o) -> (b, 1, 1, n, o) -> (b, n_s, n, n, o)
        obs = obs.contiguous().view(batch_size, self.sample_size, self.n_, self.n_*self.obs_dim) # shape = (b, n_s, n, n, o) -> (b, n_s, n, n*o)
        inp = torch.cat((obs, act), dim=-1) # shape = (b, n_s, n, n*o+n*a)

        inp = inp.contiguous().view(batch_size*self.sample_size, self.n_, -1) # shape = (b*n_s, n, n*o+n*a)

        # add agent id
        agent_ids = torch.eye(self.n_).unsqueeze(0).repeat(batch_size*self.sample_size, 1, 1) # shape = (b*n_s, n, n)
        agent_ids = cuda_wrapper(agent_ids, self.cuda_)
        inp = torch.cat( (inp, agent_ids), dim=-1 ) # shape = (b, n, n*o+n)

        # inputs = inp.contiguous().view( -1, self.n_ * (self.obs_dim + self.act_dim + 1) ) # shape = (-1, n*o+n*a+n)
        inputs = inp.contiguous().view( batch_size*self.sample_size*self.n_, -1 ) # shape = (b*n_s*n, n*o+n*a)
        agent_value = self.value_dicts[0]
        values, _ = agent_value(inputs, None)
        values = values.contiguous().view(batch_size, self.sample_size, self.n_, 1) # shape = (b, n_s, n, 1)

        return values

    def value(self, obs, act):
        return self.marginal_contribution(obs, act)
        
    def get_actions(self, state, status, exploration, actions_avail, target=False):
        if self.args.continuous:
            means, log_stds, _ = self.policy(state) if not target else self.target_net.policy(state)
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
            logits, _, _ = self.policy(state) if not target else self.target_net.policy(state)
            logits[actions_avail == 0] = -9999999
            actions, log_prob_a = select_action(self.args, logits, status=status, exploration=exploration)
            restore_actions = actions
            action_out = logits
        return actions, restore_actions, log_prob_a, action_out

    def get_loss(self, batch):
        batch_size = len(batch.state)
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, next_state, done, last_step, actions_avail = self.unpack_data(batch)
        _, actions_pol, log_prob_a, action_out = self.get_actions(state, status='train', exploration=False, actions_avail=actions_avail, target=False)
        _, next_actions, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=self.args.target)
        shapley_values_pol = self.marginal_contribution(state, actions_pol).mean(dim=1).contiguous().view(-1, self.n_)
        # do the exploration action on the value loss
        shapley_values_sum = self.marginal_contribution(state, actions).mean(dim=1).contiguous().view(-1, self.n_).sum(dim=-1, keepdim=True).expand(batch_size, self.n_)
        if self.args.target:
            next_shapley_values_sum = self.target_net.marginal_contribution(next_state, next_actions.detach()).mean(dim=1).contiguous().view(-1, self.n_).sum(dim=-1, keepdim=True).expand(batch_size, self.n_)
        else:
            next_shapley_values_sum = self.marginal_contribution(next_state, next_actions.detach()).mean(dim=1).contiguous().view(-1, self.n_).sum(dim=-1, keepdim=True).expand(batch_size, self.n_)
        returns = cuda_wrapper(torch.zeros((batch_size, self.n_), dtype=torch.float), self.cuda_)
        assert shapley_values_sum.size() == next_shapley_values_sum.size()
        assert returns.size() == shapley_values_sum.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_shapley_values_sum[i].detach()
            else:
                next_return = next_shapley_values_sum[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return
        deltas = returns - shapley_values_sum
        advantages = shapley_values_pol
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        policy_loss = - advantages
        # policy_loss = policy_loss.mean(dim=0)
        # value_loss = deltas.pow(2).mean(dim=0)
        policy_loss = policy_loss.mean()
        value_loss = deltas.pow(2).mean()
        return policy_loss, value_loss, action_out
