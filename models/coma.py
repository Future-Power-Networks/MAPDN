import torch as th
import torch.nn as nn
import numpy as np
from utilities.util import select_action, multinomials_log_density, normal_log_density
from models.model import Model
from critics.mlp_critic import MLPCritic
import torch.nn as nn



class COMA(Model):
    def __init__(self, args, target_net=None):
        super(COMA, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)

    def construct_value_net(self):
        if self.args.continuous:
            if self.args.agent_id:
                input_shape = (self.n_ + 1) * self.obs_dim + self.n_ * self.act_dim + self.n_
            else:
                input_shape = (self.n_ + 1) * self.obs_dim + self.n_ * self.act_dim
            output_shape = 1
        else:
            if self.args.agent_id:
                input_shape = (self.n_ + 1) * self.obs_dim + self.n_ * self.act_dim + self.n_
            else:
                input_shape = (self.n_ + 1) * self.obs_dim + self.n_ * self.act_dim
            output_shape = self.act_dim
        if self.args.shared_params:
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) ] )
        else:
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args) for _ in range(self.n_) ] )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def value(self, obs, act):
        gaussian_flag = (obs.size(0) != act.size(0))
        if gaussian_flag:
            batch_size = obs.size(0)
            obs_own = obs.clone()
            obs_own = obs_own.unsqueeze(0).expand(self.sample_size, batch_size, self.n_, self.obs_dim).contiguous().view(-1, self.n_, self.obs_dim) # (s*b, n, o)
            obs = obs.unsqueeze(0).unsqueeze(2).expand(self.sample_size, batch_size, self.n_, self.n_, self.obs_dim) # shape = (b, n, o) -> (1, b, 1, n, o) -> (s, b, n, n, o)
            obs = obs.contiguous().view(-1, self.n_, self.n_*self.obs_dim) # (s*b, n, n*o)
            inp = th.cat( (obs, obs_own), dim=-1 ) # shape = (s*b, n, o*n+o)
        else:
            batch_size = obs.size(0)
            obs_own = obs.clone()
            obs = obs.unsqueeze(1).expand(batch_size, self.n_, self.n_, self.obs_dim) # shape = (b, n, o) -> (b, 1, n, o) -> (b, n, n, o)
            obs = obs.contiguous().view(batch_size, self.n_, -1) # shape = (b, n, o*n)
            inp = th.cat((obs, obs_own), dim=-1) # shape = (b, n, o*n+o)

        # add agent id
        if self.args.agent_id:
            agent_ids = th.eye(self.n_).unsqueeze(0).repeat(inp.size(0), 1, 1).to(self.device) # shape = (b/s*b, n, n)
            inp = th.cat( (inp, agent_ids), dim=-1 ) # shape = (b/s*b, n, o*n+o+n)
            if self.args.shared_params:
                inp = inp.contiguous().view( -1, self.obs_dim*(self.n_+1)+self.n_ ) # shape = (b*n/s*b*n, o*n+o+n)
        else:
            if self.args.shared_params:
                inp = inp.contiguous().view( -1, self.obs_dim*(self.n_+1) ) # shape = (b*n/s*b*n, o*n+o)

        if self.args.continuous:
            if gaussian_flag:
                if self.args.shared_params:
                    act = act.contiguous().view(-1, self.n_*self.act_dim)
                inputs = th.cat( (inp, act), dim=-1 )
            else:
                act_reshape = act.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, a)
                if self.args.shared_params:
                    act_reshape = act_reshape.contiguous().view(-1, self.n_*self.act_dim)
                else:
                    act_reshape = act_reshape.contiguous().view(batch_size, self.n_, self.n_*self.act_dim) # shape = (b, n, n*a)
                inputs = th.cat( (inp, act_reshape), dim=-1 )
        else:
            # other agents' actions
            act_repeat = act.unsuqeeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, a)
            agent_mask = th.eye(self.n_).unsqueeze(0).unsqueeze(-1).expand_as(act_repeat).to(self.device) # shape = (b, n, n, a)
            agent_mask_complement = 1. - agent_mask
            act_mask_out = agent_mask_complement * act_repeat # shape = (b, n, n, a)
            if self.args.shared_params:
                act_other = act_mask_out.contiguous().view(-1, self.n_*self.act_dim) # shape = (b*n, n*a)
            else:
                act_other = act_mask_out.contiguous().view(batch_size, self.n_, self.n_*self.act_dim) # shape = (b, n, n*a)
            inputs = th.cat( (inp, act_other), dim=-1 )

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
        if self.args.continuous:
            means, log_stds, _ = self.policy(state, last_hid=last_hids)
            action_out = (means, log_stds)
            log_prob_a = normal_log_density(actions, means, log_stds)
            if self.args.double_q:
                _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=hids)
            else:
                _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=True, last_hid=hids)
            self.sample_size = self.args.sample_size
            means, log_stds = action_out
            means_repeat = means.unsqueeze(0).repeat(self.sample_size, 1, 1, 1) # (s,b,n,a)
            log_stds_repeat = log_stds.unsqueeze(0).repeat(self.sample_size, 1, 1, 1) # (s,b,n,a)
            _sampled_actions_repeat = th.normal(means_repeat, log_stds_repeat.exp()) # (s,b,n,a)
            sampled_actions_repeat = _sampled_actions_repeat.unsqueeze(2).repeat(1, 1, self.n_, 1, 1) # (s,b,n,a) -> (s,b,n,n,a)
            actions_repeat = actions.unsqueeze(0).unsqueeze(2).repeat(self.sample_size, 1, self.n_, 1, 1) # (s,b,n,n,a)
            agent_mask = th.eye(self.n_).unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand_as(actions_repeat).to(self.device) # (s,b,n,n,a)
            agent_mask_complement = 1. - agent_mask
            actions_repeat_merge = actions_repeat * agent_mask_complement + sampled_actions_repeat * agent_mask # (s,b,n,n,a)
            actions_repeat_merge = actions_repeat_merge.contiguous().view(-1, self.n_, self.n_*self.act_dim) # (s*b,n,n*a)
            values_sampled = self.value(state, actions_repeat_merge).contiguous().view(self.sample_size, batch_size, self.n_) # (s*b,n,1) -> (s,b,n)
            baselines = th.mean(values_sampled, dim=0) # (b,n)
            values = self.value(state, actions).squeeze(-1) # (b,n,a) -> (b,n) action value
        else:
            logits, _ = self.policy(state, last_hid=last_hids)
            action_out = logits
            log_prob_a = multinomials_log_density(actions, logits)
            if self.args.double_q:
                _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=hids)
            else:
                _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=True, last_hid=hids)
            values = self.value(state, actions) # (b,n,a) action value
            baselines = th.sum(values*th.softmax(logits, dim=-1), dim=-1) # the only difference to ActorCritic is this  baseline (b,n)
            values = th.sum(values*actions, dim=-1) # (b,n)
        if self.args.target:
            next_values = self.target_net.value(next_state, next_actions)
        else:
            next_values = self.value(next_state, next_actions)
        if self.args.continuous:
            next_values = next_values.squeeze(-1) # (b,n)
        else:
            next_values = th.sum(next_values*next_actions, dim=-1) # (b,n)
        # calculate the advantages
        returns = th.zeros((batch_size, self.n_), dtype=th.float).to(self.device)
        assert values.size() == next_values.size()
        assert returns.size() == values.size()
        done = done.to(self.device)
        returns = rewards + self.args.gamma * (1 - done) * next_values.detach()
        # value loss
        deltas = returns - values
        value_loss = deltas.pow(2).mean()
        # actio loss
        advantages = ( values - baselines ).detach()
        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)
        restore_mask = 1. - (actions_avail == 0).to(self.device).float()
        log_prob_a = (restore_mask * log_prob_a).sum(dim=-1)
        log_prob_a = log_prob_a.squeeze(-1)
        assert log_prob_a.size() == advantages.size(), f"log_prob size is: {log_prob_a.size()} and advantages size is {advantages.size()}."
        policy_loss = - advantages * log_prob_a
        policy_loss = policy_loss.mean()
        return policy_loss, value_loss, action_out
