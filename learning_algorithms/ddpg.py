from learning_algorithms.rl_algorithms import ReinforcementLearning
import torch as th
import torch.nn as nn



class DDPG(ReinforcementLearning):
    def __init__(self, args):
        super(DDPG, self).__init__('DDPG', args)
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)

    def __call__(self, batch, behaviour_net, target_net):
        return self.get_loss(batch, behaviour_net, target_net)

    def get_loss(self, batch, behaviour_net, target_net):
        batch_size = len(batch.state)
        n = self.args.agent_num
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, next_state, done, last_step, actions_avail, last_hids, hids = behaviour_net.unpack_data(batch)
        _, actions_pol, log_prob_a, action_out, _ = behaviour_net.get_actions(state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=last_hids)
        if self.args.double_q:
            _, next_actions, _, _, _ = behaviour_net.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=hids)
        else:
            _, next_actions, _, _, _ = behaviour_net.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=True, last_hid=hids)
        values_pol = behaviour_net.value(state, actions_pol).contiguous().view(-1, n)
        values = behaviour_net.value(state, actions).contiguous().view(-1, n)
        next_values = target_net.value(next_state, next_actions.detach()).contiguous().view(-1, n)
        returns = th.zeros( (batch_size, n), dtype=th.float ).to(self.device)
        assert values.size() == next_values.size()
        assert returns.size() == values.size()
        done = done.to(self.device)
        returns = rewards + self.args.gamma * (1 - done) * next_values.detach()
        deltas = returns - values
        advantages = values_pol
        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)
        action_loss = - advantages
        action_loss = action_loss.mean()
        value_loss = deltas.pow(2).mean()
        return action_loss, value_loss, action_out
