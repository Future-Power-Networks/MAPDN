import torch as th
import torch.nn as nn
import numpy as np
from utilities.util import select_action
from models.model import Model
from critics.mlp_critic import MLPCritic



class MATD3(Model):
    def __init__(self, args, target_net=None):
        super(MATD3, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)

    def construct_value_net(self):
        if self.args.agent_id:
            input_shape = (self.obs_dim + self.act_dim) * self.n_ + 1 + self.n_
        else:
            input_shape = (self.obs_dim + self.act_dim) * self.n_ + 1
        output_shape = 1
        if self.args.shared_params:
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) ] )
        else:
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) for _ in range(self.n_) ] )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def value(self, obs, act):
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)
        batch_size = obs.size(0)

        obs_repeat = obs.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, o)
        obs_reshape = obs_repeat.contiguous().view(batch_size, self.n_, -1) # shape = (b, n, n*o)

        # add agent id
        agent_ids = th.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device) # shape = (b, n, n)
        if self.args.agent_id:
            obs_reshape = th.cat( (obs_reshape, agent_ids), dim=-1 ) # shape = (b, n, n*o+n)

        act_repeat = act.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, a)
        act_mask_others = agent_ids.unsqueeze(-1) # shape = (b, n, n, 1)
        act_mask_i = 1. - act_mask_others
        act_i = act_repeat * act_mask_others
        act_others = act_repeat * act_mask_i

        # detach other agents' actions
        act_repeat = act_others.detach() + act_i # shape = (b, n, n, a)
        
        if self.args.shared_params:
            obs_reshape = obs_reshape.contiguous().view( batch_size*self.n_, -1 ) # shape = (b*n, n*o+n/n*o)
            act_reshape = act_repeat.contiguous().view( batch_size*self.n_, -1 ) # shape = (b*n, n*a)
        else:
            obs_reshape = obs_reshape.contiguous().view( batch_size, self.n_, -1 ) # shape = (b, n, n*o+n/n*o)
            act_reshape = act_repeat.contiguous().view( batch_size, self.n_, -1 ) # shape = (b, n, n*a)

        inputs = th.cat( (obs_reshape, act_reshape), dim=-1 )
        ones = th.ones( inputs.size()[:-1] + (1,), dtype=th.float ).to(self.device)
        zeros = th.zeros( inputs.size()[:-1] + (1,), dtype=th.float ).to(self.device)
        inputs1 = th.cat( (inputs, zeros), dim=-1 )
        inputs2 = th.cat( (inputs, ones), dim=-1 )

        if self.args.shared_params:
            agent_value = self.value_dicts[0]
            values1, _ = agent_value(inputs1, None)
            values2, _ = agent_value(inputs2, None)
            values1 = values1.contiguous().view(batch_size, self.n_, 1)
            values2 = values2.contiguous().view(batch_size, self.n_, 1)
        else:
            values1, values2 = [], []
            for i, agent_value in enumerate(self.value_dicts):
                values_1, _ = agent_value(inputs1[:, i, :], None)
                values_2, _ = agent_value(inputs2[:, i, :], None)
                values1.append(values_1)
                values2.append(values_2)
            values1 = th.stack(values1, dim=1)
            values2 = th.stack(values2, dim=1)

        return th.cat([values1, values2], dim=0)

    def get_actions(self, obs, status, exploration, actions_avail, target=False, last_hid=None, clip=False):
        target_policy = self.target_net.policy if self.args.target else self.policy
        if self.args.continuous:
            means, log_stds, hiddens = self.policy(obs, last_hid=last_hid) if not target else target_policy(obs, last_hid=last_hid)
            means[actions_avail == 0] = 0.0
            log_stds[actions_avail == 0] = 0.0
            if means.size(-1) > 1:
                means_ = means.sum(dim=1, keepdim=True)
                log_stds_ = log_stds.sum(dim=1, keepdim=True)
            else:
                means_ = means
                log_stds_ = log_stds
            actions, log_prob_a = select_action(self.args, means_, status=status, exploration=exploration, info={'clip': clip, 'log_std': log_stds_})
            restore_mask = 1. - (actions_avail == 0).to(self.device).float()
            restore_actions = restore_mask * actions
            action_out = (means, log_stds)
        else:
            logits, _, hiddens = self.policy(obs, last_hid=last_hid) if not target else target_policy(obs, last_hid=last_hid)
            logits[actions_avail == 0] = -9999999
            # this follows the original version of sac: sampling actions
            actions, log_prob_a = select_action(self.args, logits, status=status, exploration=exploration)
            restore_actions = actions
            action_out = logits
        return actions, restore_actions, log_prob_a, action_out, hiddens

    def get_loss(self, batch):
        batch_size = len(batch.state)
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, next_state, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)
        _, actions_pol, log_prob_a, action_out, _ = self.get_actions(state, status='train', exploration=False, \
            actions_avail=actions_avail, target=False, last_hid=last_hids)
        # _, next_actions, _, _, _ = self.get_actions(next_obs, status='train', exploration=True, actions_avail=actions_avail, target=True, last_hid=hids)
        if self.args.double_q:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=True, \
                actions_avail=actions_avail, target=False, last_hid=hids, clip=True)
        else:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=True, \
                actions_avail=actions_avail, target=True, last_hid=hids, clip=True)
        compose_pol = self.value(state, actions_pol)
        values_pol = compose_pol[:batch_size, :]
        values_pol = values_pol.contiguous().view(-1, self.n_)
        compose = self.value(state, actions)
        values1, values2 = compose[:batch_size, :], compose[batch_size:, :]
        values1 = values1.contiguous().view(-1, self.n_)
        values2 = values2.contiguous().view(-1, self.n_)
        next_compose = self.target_net.value(next_state, next_actions.detach())
        next_values1, next_values2 = next_compose[:batch_size, :], next_compose[batch_size:, :]
        next_values1 = next_values1.contiguous().view(-1, self.n_)
        next_values2 = next_values2.contiguous().view(-1, self.n_)
        returns = th.zeros((batch_size, self.n_), dtype=th.float, device=self.device)
        assert values_pol.size() == next_values1.size() == next_values2.size()
        assert returns.size() == values1.size() == values2.size()
        # update twin values by the minimized target q
        done = done.to(self.device)
        next_values = th.stack([next_values1, next_values2], -1)
        returns = rewards + self.args.gamma * (1 - done) * th.min(next_values.detach(), -1)[0]
        deltas1 = returns - values1
        deltas2 = returns - values2
        advantages = values_pol
        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)
        policy_loss = - advantages
        policy_loss = policy_loss.mean()
        value_loss = 0.5 * ( deltas1.pow(2).mean() + deltas2.pow(2).mean() )
        return policy_loss, value_loss, action_out
