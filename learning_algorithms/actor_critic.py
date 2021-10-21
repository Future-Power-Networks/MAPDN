from learning_algorithms.rl_algorithms import ReinforcementLearning
import torch as th
import torch.nn as nn
from utilities.util import multinomials_log_density, normal_log_density



class ActorCritic(ReinforcementLearning):
    def __init__(self, args):
        super(ActorCritic, self).__init__('Actor_Critic', args)
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)

    def __call__(self, batch, behaviour_net):
        return self.get_loss(batch, behaviour_net)

    def get_loss(self, batch, behaviour_net, target_net=None):
        batch_size = len(batch.state)
        n = self.args.agent_num
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, next_state, done, last_step, actions_avail, last_hids, hids = behaviour_net.unpack_data(batch)
        if self.args.continuous:
            means, log_stds, _ = behaviour_net.policy(state, last_hid=last_hids)
            action_out = (means, log_stds)
            log_prob_a = normal_log_density(actions, means, log_stds)
            restore_mask = 1. - (actions_avail == 0).to(self.device).float()
            log_prob_a = (restore_mask * log_prob_a).sum(dim=-1)
        else:
            logits, _ = behaviour_net.policy(state, last_hid=last_hids)
            action_out = logits
            log_prob_a = multinomials_log_density(actions, logits)
        if self.args.double_q:
            _, next_actions, _, _, _ = behaviour_net.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=hids)
        else:
            _, next_actions, _, _, _ = behaviour_net.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=True, last_hid=hids)
        values = behaviour_net.value(state, actions)
        if not self.args.continuous:
            values = th.sum(values*actions, dim=-1)
        values = values.contiguous().view(-1, n)
        next_values = behaviour_net.value(next_state, next_actions.detach())
        if not self.args.continuous:
            next_values = th.sum(next_values*next_actions.detach(), dim=-1)
        next_values = next_values.contiguous().view(-1, n)
        returns = th.zeros( (batch_size, n), dtype=th.float ).to(behaviour_net.device)
        assert values.size() == next_values.size()
        assert returns.size() == values.size()
        done = done.to(self.device)
        returns = rewards + self.args.gamma * (1 - done) * next_values.detach()
        deltas = returns - values
        advantages = values.detach()
        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)
        log_prob_a = log_prob_a.squeeze(-1)
        assert log_prob_a.size() == advantages.size(), f"This is the log_prob_a size: {log_prob_a.size()} and advantage size: {advantages.size()}"
        action_loss = - advantages * log_prob_a
        action_loss = action_loss.mean()
        value_loss = deltas.pow(2).mean()
        return action_loss, value_loss, action_out
