import torch
import torch.nn as nn
import numpy as np
from utilities.util import select_action, cuda_wrapper, batchnorm, multinomials_log_density, normal_log_density
from models.model import Model
from collections import namedtuple
from critics.maac_critic import AttentionCritic


class MAAC(Model):
    def __init__(self, args, target_net=None):
        super(MAAC, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()

    def construct_policy_net(self):
        if self.args.agent_id:
            input_shape = self.obs_dim + self.n_
        else:
            input_shape = self.obs_dim

        if self.args.agent_type == 'mlp':
            from agents.mlp_agent_gaussian import MLPAgent
            Agent = MLPAgent
        elif self.args.agent_type == 'rnn':
            from agents.rnn_agent_gaussian import RNNAgent
            Agent = RNNAgent
        else:
            NotImplementedError()
            
        if self.args.shared_params:
            self.policy_dicts = nn.ModuleList([ Agent(input_shape, self.args) ])
        else:
            self.policy_dicts = nn.ModuleList([ Agent(input_shape, self.args) for _ in range(self.n_) ])

    def construct_value_net(self):
        self.value_dicts = nn.ModuleList( [ AttentionCritic(self.args) ] )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        # obs_shape = (b, n, o)
        batch_size = obs.size(0)

        # add agent id
        if self.args.agent_id:
            agent_ids = torch.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1) # shape = (b, n, n)
            agent_ids = cuda_wrapper(agent_ids, self.cuda_)
            obs = torch.cat( (obs, agent_ids), dim=-1 ) # shape = (b, n, o+n)

        if self.args.shared_params:
            # print (f"This is the shape of last_hids: {last_hid.size()}")
            obs = obs.contiguous().view(batch_size*self.n_, -1) # shape = (b*n, n+o/o)
            agent_policy = self.policy_dicts[0]
            means, log_stds, hiddens = agent_policy(obs, last_hid)
            # hiddens = torch.stack(hiddens, dim=1)
            means = means.contiguous().view(batch_size, self.n_, -1)
            hiddens = hiddens.contiguous().view(batch_size, self.n_, -1)
            log_stds = log_stds.contiguous().view(batch_size, self.n_, -1)
        else:
            means = []
            hiddens = []
            log_stds = []
            for i, agent_policy in enumerate(self.policy_dicts):
                mean, log_std, hidden = agent_policy(obs[:, i, :], last_hid[:, i, :])
                means.append(mean)
                hiddens.append(hidden)
                log_stds.append(log_std)
            means = torch.stack(means, dim=1)
            hiddens = torch.stack(hiddens, dim=1)
            log_stds = torch.stack(log_stds, dim=1)

        return means, log_stds, hiddens

    def value(self, obs, act):
        '''
        refer to the implementation of MAAC from https://github.com/shariqiqbal2810/MAAC.
        '''
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)

        obs_chunks = [chk.squeeze(1) for chk in torch.chunk(obs, self.n_, dim=1)]
        act_chunks = [chk.squeeze(1) for chk in torch.chunk(act, self.n_, dim=1)]
        
        inps_chk = torch.cat( (obs, act), dim=-1 ) 
        sa_chunks = [chk.squeeze(1) for chk in torch.chunk(inps_chk, self.n_, dim=1)]
        inps = (obs_chunks, act_chunks, sa_chunks)
        agents_rets = self.value_dicts[0](inps)
        
        dec_agents_rets = []
        for val in zip(*agents_rets):
            dec_agents_rets.append(torch.cat(val, dim=1))

        return torch.cat(dec_agents_rets, dim=0)

    def get_actions(self, state, status, exploration, actions_avail, target=False, last_hid=None):
        if self.args.continuous:
            means, log_stds, hiddens = self.policy(state, last_hid=last_hid) if not target else self.target_net.policy(state, last_hid=last_hid)
            if means.size(-1) > 1:
                means_ = means.sum(dim=1, keepdim=True)
                log_stds_ = log_stds.sum(dim=1, keepdim=True)
            else:
                means_ = means
                log_stds_ = log_stds
            # this follows the original version of sac: sampling actions
            actions, log_prob_a = select_action(self.args, means_, status=status, exploration=exploration, info={'enforcing_action_bound': self.args.action_enforcebound, 'log_std': log_stds_})
            restore_mask = 1. - cuda_wrapper((actions_avail == 0).float(), self.cuda_)
            if log_prob_a != None:
                log_prob_a = (restore_mask * log_prob_a).sum(dim=-1)
            restore_actions = restore_mask * actions
            action_out = (means, log_stds)
        else:
            logits, _, hiddens = self.policy(state, last_hid=last_hid) if not target else self.target_net.policy(state, last_hid=last_hid)
            logits[actions_avail == 0] = -9999999
            # this follows the original version of sac: sampling actions
            actions, log_prob_a = select_action(self.args, logits, status=status, exploration=exploration)
            restore_actions = actions
            action_out = logits
        return actions, restore_actions, log_prob_a, action_out, hiddens

    def get_loss(self, batch):
        batch_size = len(batch.state)
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, next_state, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)
        _, actions_pol, log_prob_a, action_out, _ = self.get_actions(state, status='train', exploration=True, actions_avail=actions_avail, target=False, last_hid=last_hids)
        _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=True, actions_avail=actions_avail, target=self.args.target, last_hid=hids)
        values_pol = self.value(state, actions_pol)[:batch_size, :]
        compose = self.value(state, actions.detach()) # values = (b. n) / attn_reg = (1, n)
        values, attn_reg = compose[:batch_size, :], compose[batch_size:batch_size+1, :]
        next_values = self.target_net.value(next_state, next_actions.detach())[:batch_size, :]
        next_values = next_values.contiguous().view(-1, self.n_)
        returns = cuda_wrapper(torch.zeros((batch_size, self.n_), dtype=torch.float), self.cuda_)
        assert values.size() == next_values.size()
        assert returns.size() == values.size()
        assert returns.size() == log_prob_a.size(), f"returns size: {returns.size()} and log_prob_a size: {log_prob_a.size()}"
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_values[i].detach()
            else:
                next_return = next_values[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return
            if self.args.soft:
                returns[i] -= log_prob_a[i] / self.args.reward_scale
        deltas = returns - values
        advantages = values_pol
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        if self.args.soft:
            policy_loss = log_prob_a / self.args.reward_scale - advantages
        else:
            policy_loss = - advantages.detach() * log_prob_a
        policy_loss += attn_reg.squeeze(0)
        policy_loss = policy_loss.mean()
        value_loss = deltas.pow(2).mean()
        return policy_loss, value_loss, action_out
            