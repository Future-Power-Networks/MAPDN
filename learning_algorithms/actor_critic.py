from learning_algorithms.rl_algorithms import ReinforcementLearning
import torch
from utilities.util import select_action, cuda_wrapper, batchnorm, multinomials_log_density, normal_log_density


class ActorCritic(ReinforcementLearning):
    def __init__(self, args):
        super(ActorCritic, self).__init__('Actor_Critic', args)

    def __call__(self, batch, behaviour_net):
        return self.get_loss(batch, behaviour_net)

    def get_loss(self, batch, behaviour_net, target_net=None):
        batch_size = len(batch.state)
        n = self.args.agent_num
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, next_state, done, last_step, actions_avail = behaviour_net.unpack_data(batch)
        if self.args.continuous:
            means, log_stds, _ = behaviour_net.policy(state)
            action_out = (means, log_stds)
            log_prob_a = normal_log_density(actions, means, log_stds)
            restore_mask = 1. - cuda_wrapper((actions_avail == 0).float(), self.cuda_)
            log_prob_a = (restore_mask * log_prob_a).sum(dim=-1)
        else:
            logits, _ = behaviour_net.policy(state)
            action_out = logits
            log_prob_a = multinomials_log_density(actions, logits)
        _, next_actions, _, _ = behaviour_net.get_actions(next_state, status='train', exploration=True, actions_avail=actions_avail, target=self.args.target)
        # values_pol = behaviour_net.value(state, actions)
        values = behaviour_net.value(state, actions)
        if not self.args.continuous:
            values = torch.sum(values*actions, dim=-1)
        values = values.contiguous().view(-1, n)
        next_values = behaviour_net.value(next_state, next_actions.detach())
        if not self.args.continuous:
            next_values = torch.sum(next_values*next_actions.detach(), dim=-1)
        next_values = next_values.contiguous().view(-1, n)
        returns = cuda_wrapper( torch.zeros((batch_size, n), dtype=torch.float), self.cuda_ )
        assert values.size() == next_values.size()
        assert returns.size() == values.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_values[i].detach()
            else:
                next_return = next_values[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return
        deltas = returns - values
        advantages = values.detach()
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        log_prob_a = log_prob_a.squeeze(-1)
        assert log_prob_a.size() == advantages.size(), f"This is the log_prob_a size: {log_prob_a.size()} and advantage size: {advantages.size()}"
        action_loss = - advantages * log_prob_a
        # action_loss = action_loss.mean(dim=0)
        # value_loss = deltas.pow(2).mean(dim=0)
        action_loss = action_loss.mean()
        value_loss = deltas.pow(2).mean()
        return action_loss, value_loss, action_out
