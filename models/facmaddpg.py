import torch as th
import torch.nn as nn
import numpy as np
from utilities.util import select_action
from models.model import Model
from critics.mlp_critic import MLPCritic
from critics.qmix import QMixer



class FACMADDPG(Model):
    def __init__(self, args, target_net=None):
        super(FACMADDPG, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)

    def construct_value_net(self):
        if self.args.agent_id:
            input_shape = self.obs_dim + self.act_dim + self.n_
        else:
            input_shape = self.obs_dim + self.act_dim
        output_shape = 1
        if self.args.shared_params:
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) ] )
        else:
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) for _ in range(self.n_) ] )
        # self.value_dicts.append(self.mixer)
        self.mixer = QMixer(self.args)

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def value(self, obs, act):
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
            agent_value = self.value_dicts[0]
            inputs = th.cat([obs, act], dim=-1)
            values, _ = agent_value(inputs, None)
            values = values.contiguous().view(batch_size, self.n_, -1)
        else:
            inputs = th.cat([obs, act], dim=-1) # shape = (b, n, o+a+n/o+a)
            values = []
            for i, agent_value in enumerate(self.value_dicts):
                value, _ = agent_value(inputs[:, i, :], None)
                values.append(value)
            values = th.stack(values, dim=1)

        return values

    def get_actions(self, state, status, exploration, actions_avail, target=False, last_hid=None):
        target_policy = self.target_net.policy if self.args.target else self.policy
        if self.args.continuous:
            means, log_stds, hiddens = self.policy(state, last_hid=last_hid) if not target else target_policy(state, last_hid=last_hid)
            if means.size(-1) > 1:
                means_ = means.sum(dim=1, keepdim=True)
                log_stds_ = log_stds.sum(dim=1, keepdim=True)
            else:
                means_ = means
                log_stds_ = log_stds
            actions, log_prob_a = select_action(self.args, means_, status=status, exploration=exploration, info={'log_std': log_stds_})
            restore_mask = 1. - (actions_avail == 0).to(self.device).float()
            restore_actions = restore_mask * actions
            action_out = (means, log_stds)
        else:
            logits, _, hiddens = self.policy(state, last_hid=last_hid) if not target else target_policy(state, last_hid=last_hid)
            logits[actions_avail == 0] = -9999999
            actions, log_prob_a = select_action(self.args, logits, status=status, exploration=exploration)
            restore_actions = actions
            action_out = logits
        return actions, restore_actions, log_prob_a, action_out, hiddens

    def get_loss(self, batch):
        batch_size = len(batch.state)
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, next_state, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)
        _, actions_pol, log_prob_a, action_out, _ = self.get_actions(state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=last_hids)
        if self.args.double_q:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=hids)
        else:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=True, last_hid=hids)
        values_pol = self.value(state, actions_pol).contiguous().view(-1, self.n_) # shape = (b, n)
        # get the q_tot
        values = self.value(state, actions).contiguous().view(-1, self.n_) # shape = (b, n)
        global_state = state.contiguous().view(batch_size, self.n_*self.obs_dim) # shape = (b, n*o)
        q_tot = self.mixer(values, global_state).contiguous().view(-1, 1) # shape = (b, 1)
        # get the next q_tot
        next_values = self.target_net.value(next_state, next_actions.detach()).contiguous().view(-1, self.n_) # shape = (b, n)
        next_global_state = next_state.contiguous().view(batch_size, self.n_*self.obs_dim) # shape = (b, n*o)
        next_q_tot = self.target_net.mixer(next_values, next_global_state).contiguous().view(-1, 1) # shape = (b, 1)
        returns = th.zeros_like(q_tot).to(self.device)
        assert q_tot.size() == next_q_tot.size()
        assert returns.size() == q_tot.size()
        done = done.to(self.device)
        returns = rewards[:, 0:1] + self.args.gamma * (1 - done) * next_q_tot.detach()
        deltas = returns - q_tot
        advantages = values_pol
        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)
        policy_loss = - advantages
        policy_loss = policy_loss.mean()
        value_loss = deltas.pow(2).mean()
        return policy_loss, value_loss, action_out
