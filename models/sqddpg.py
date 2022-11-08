import torch as th
import torch.nn as nn
import numpy as np
from utilities.util import select_action
from models.model import Model
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
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)
        self.agent_importance_vec = th.tensor(np.ones(self.n_)/self.n_, device=self.device)

    def construct_value_net(self):
        if self.args.agent_id:
            input_shape = (self.obs_dim + self.act_dim) * self.n_ + self.n_
        else:
            input_shape = (self.obs_dim + self.act_dim) * self.n_
        output_shape = 1
        if self.args.shared_params:
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) ] )
        else:
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) for _ in range(self.n_) ] )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def sample_grandcoalitions(self, batch_size):
        seq_set = th.tril(th.ones(self.n_, self.n_, device=self.device), diagonal=0, out=None)
        agent_importance_vec = self.agent_importance_vec.unsqueeze(0).expand(batch_size*self.sample_size, self.n_)
        grand_coalitions_pos = th.multinomial(agent_importance_vec, self.n_, replacement=False) # shape = (b*n_s, n)
        individual_map = th.zeros((batch_size*self.sample_size*self.n_, self.n_), device=self.device)
        individual_map.scatter_(1, grand_coalitions_pos.contiguous().view(-1, 1), 1)
        individual_map = individual_map.contiguous().view(batch_size, self.sample_size, self.n_, self.n_)
        subcoalition_map = th.matmul(individual_map, seq_set)

        # FIX: construct the grand coalition (in sequence by agent_idx) from the grand_coalitions_pos (e.g., pos_idx <- grand_coalitions_pos[agent_idx])
        # grand_coalitions = []
        # for grand_coalition_pos in grand_coalitions_pos:
        #     grand_coalition = th.zeros_like(grand_coalition_pos)
        #     for agent, pos in enumerate(grand_coalition_pos):
        #         grand_coalition[pos] = agent
        #     grand_coalitions.append(grand_coalition)
        # grand_coalitions = th.stack(grand_coalitions, dim=0).to(self.device)
        offset = (th.arange(batch_size*self.sample_size, device=self.device)*self.n_).reshape(-1, 1)
        grand_coalitions_pos_alter = grand_coalitions_pos + offset
        grand_coalitions = th.zeros_like(grand_coalitions_pos_alter.flatten(), device=self.device)
        grand_coalitions[grand_coalitions_pos_alter.flatten()] = th.arange(batch_size*self.sample_size*self.n_, device=self.device)
        grand_coalitions = grand_coalitions.reshape(batch_size*self.sample_size, self.n_) - offset

        grand_coalitions = grand_coalitions.unsqueeze(1).expand(batch_size*self.sample_size, \
            self.n_, self.n_).contiguous().view(batch_size, self.sample_size, self.n_, self.n_) # shape = (b, n_s, n, n)

        return subcoalition_map, grand_coalitions, individual_map

    def marginal_contribution(self, obs, act):
        batch_size = obs.size(0)
        subcoalition_map, grand_coalitions, individual_map = self.sample_grandcoalitions(batch_size) # shape = (b, n_s, n, n)
        grand_coalitions_expand = grand_coalitions.unsqueeze(-1).expand(batch_size, self.sample_size, self.n_, self.n_, self.act_dim) # shape = (b, n_s, n, n, a)
        act = act.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_, self.n_, self.act_dim).gather(3, grand_coalitions_expand) # shape = (b, n, a) -> (b, 1, 1, n, a) -> (b, n_s, n, n, a)
        subcoalition_map_no_i = subcoalition_map - individual_map # shape = (b, n_s, n, n)
        subcoalition_map_no_i = subcoalition_map_no_i.unsqueeze(-1) # shape = (b, n_s, n, n, 1)
        individual_map = individual_map.unsqueeze(-1) # shape = (b, n_s, n, n, 1)
        act_no_i = act * subcoalition_map_no_i
        act_i = act * individual_map

        # detach other agents' actions
        act = act_no_i.detach() + act_i # shape = (b, n_s, n, n, a)

        # make up inputs
        act = act.contiguous().view(batch_size, self.sample_size, self.n_, -1) # shape = (b, n_s, n, n*a)
        obs = obs.unsqueeze(1).unsqueeze(2).expand(batch_size, self.sample_size, self.n_, self.n_, self.obs_dim) # shape = (b, n, o) -> (b, 1, n, o) -> (b, 1, 1, n, o) -> (b, n_s, n, n, o)
        obs = obs.contiguous().view(batch_size, self.sample_size, self.n_, self.n_*self.obs_dim) # shape = (b, n_s, n, n, o) -> (b, n_s, n, n*o)
        inp = th.cat((obs, act), dim=-1) # shape = (b, n_s, n, n*o+n*a)

        inp = inp.contiguous().view(batch_size*self.sample_size, self.n_, -1) # shape = (b*n_s, n, n*o+n*a)

        # add agent id
        if self.args.agent_id:
            agent_ids = th.eye(self.n_).unsqueeze(0).repeat(batch_size*self.sample_size, 1, 1).to(self.device) # shape = (b*n_s, n, n)
            inp = th.cat( (inp, agent_ids), dim=-1 ) # shape = (b*n_s, n, n*o+n*a+n)
        
        if self.args.shared_params:
            inputs = inp.contiguous().view( batch_size*self.sample_size*self.n_, -1 ) # shape = (b*n_s*n, n*o+n*a/n*o+n*a+n)
            agent_value = self.value_dicts[0]
            values, _ = agent_value(inputs, None)
        else:
            inputs = inp
            values = []
            for i, agent_value in enumerate(self.value_dicts):
                value, _ = agent_value(inputs[:, i, :], None)
                values.append(value)
            values = th.stack(values, dim=1)
            
        values = values.contiguous().view(batch_size, self.sample_size, self.n_, 1) # shape = (b, n_s, n, 1)

        return values

    def value(self, obs, act):
        return self.marginal_contribution(obs, act)
        
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
        shapley_values_pol = self.marginal_contribution(state, actions_pol).mean(dim=1).contiguous().view(-1, self.n_)
        # do the exploration action on the value loss
        shapley_values_sum = self.marginal_contribution(state, actions).mean(dim=1).contiguous().view(-1, self.n_).sum(dim=-1, keepdim=True).expand(batch_size, self.n_)
        if self.args.target:
            next_shapley_values_sum = self.target_net.marginal_contribution(next_state, next_actions.detach()).mean(dim=1).contiguous().view(-1, self.n_).sum(dim=-1, keepdim=True).expand(batch_size, self.n_)
        else:
            next_shapley_values_sum = self.marginal_contribution(next_state, next_actions.detach()).mean(dim=1).contiguous().view(-1, self.n_).sum(dim=-1, keepdim=True).expand(batch_size, self.n_)
        returns = th.zeros((batch_size, self.n_), dtype=th.float).to(self.device)
        assert shapley_values_sum.size() == next_shapley_values_sum.size()
        assert returns.size() == shapley_values_sum.size()
        done = done.to(self.device)
        returns = rewards + self.args.gamma * (1 - done) * next_shapley_values_sum.detach()
        deltas = returns - shapley_values_sum
        advantages = shapley_values_pol
        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)
        policy_loss = - advantages
        policy_loss = policy_loss.mean()
        value_loss = deltas.pow(2).mean()
        return policy_loss, value_loss, action_out
