import torch as th
import torch.nn as nn
import numpy as np
from utilities.util import select_action
from models.model import Model
from critics.maac_critic import AttentionCritic



class MAAC(Model):
    def __init__(self, args, target_net=None):
        super(MAAC, self).__init__(args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)

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

    def value(self, obs, act):
        '''
        refer to the implementation of MAAC from https://github.com/shariqiqbal2810/MAAC.
        '''
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)

        obs_chunks = [chk.squeeze(1) for chk in th.chunk(obs, self.n_, dim=1)]
        act_chunks = [chk.squeeze(1) for chk in th.chunk(act, self.n_, dim=1)]
        
        inps_chk = th.cat( (obs, act), dim=-1 ) 
        sa_chunks = [chk.squeeze(1) for chk in th.chunk(inps_chk, self.n_, dim=1)]
        inps = (obs_chunks, act_chunks, sa_chunks)
        agents_rets = self.value_dicts[0](inps)
        
        dec_agents_rets = []
        for val in zip(*agents_rets):
            dec_agents_rets.append(th.cat(val, dim=1))

        return th.cat(dec_agents_rets, dim=0)

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
            # this follows the original version of sac: sampling actions
            actions, log_prob_a = select_action(self.args, means_, status=status, exploration=exploration, info={'log_std': log_stds_})
            restore_mask = 1. - (actions_avail == 0).to(self.device).float()
            if log_prob_a != None:
                log_prob_a = (restore_mask * log_prob_a).sum(dim=-1)
            restore_actions = restore_mask * actions
            action_out = (means, log_stds)
        else:
            logits, _, hiddens = self.policy(state, last_hid=last_hid) if not target else target_policy(state, last_hid=last_hid)
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
        _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=True, actions_avail=actions_avail, target=True, last_hid=hids)
        values_pol = self.value(state, actions_pol)[:batch_size, :]
        compose = self.value(state, actions.detach()) # values = (b. n) / attn_reg = (1, n)
        values, attn_reg = compose[:batch_size, :], compose[batch_size:batch_size+1, :]
        next_values = self.target_net.value(next_state, next_actions.detach())[:batch_size, :]
        next_values = next_values.contiguous().view(-1, self.n_)
        returns = th.zeros((batch_size, self.n_), dtype=th.float).to(self.device)
        assert values.size() == next_values.size()
        assert returns.size() == values.size()
        assert returns.size() == log_prob_a.size(), f"returns size: {returns.size()} and log_prob_a size: {log_prob_a.size()}"
        done = done.to(self.device)
        returns = rewards + self.args.gamma * (1 - done) * next_values.detach() - self.args.soft * log_prob_a / self.args.reward_scale
        deltas = returns - values
        advantages = values_pol
        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)
        if self.args.soft:
            policy_loss = log_prob_a / self.args.reward_scale - advantages
        else:
            policy_loss = - advantages.detach() * log_prob_a
        policy_loss += attn_reg.squeeze(0)
        policy_loss = policy_loss.mean()
        value_loss = deltas.pow(2).mean()
        return policy_loss, value_loss, action_out
            