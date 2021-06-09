from learning_algorithms.rl_algorithms import ReinforcementLearning
from utilities.util import select_action, cuda_wrapper, batchnorm
import torch



class DDPG(ReinforcementLearning):
    def __init__(self, args):
        super(DDPG, self).__init__('DDPG', args)

    def __call__(self, batch, behaviour_net, target_net):
        return self.get_loss(batch, behaviour_net, target_net)

    def get_loss(self, batch, behaviour_net, target_net):
        batch_size = len(batch.state)
        n = self.args.agent_num
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, next_state, done, last_step, actions_avail = behaviour_net.unpack_data(batch)
        _, actions_pol, log_prob_a, action_out = behaviour_net.get_actions(state, status='train', exploration=False, actions_avail=actions_avail, target=False)
        _, next_actions, _, _ = behaviour_net.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=self.args.target)
        values_pol = behaviour_net.value(state, actions_pol).contiguous().view(-1, n)
        values = behaviour_net.value(state, actions).contiguous().view(-1, n)
        next_values = target_net.value(next_state, next_actions.detach()).contiguous().view(-1, n)
        returns = cuda_wrapper(torch.zeros((batch_size, n), dtype=torch.float), self.cuda_)
        assert values.size() == next_values.size()
        assert returns.size() == values.size()
        for i in reversed(range(rewards.size(0))):
            if last_step[i]:
                next_return = 0 if done[i] else next_values[i].detach()
            else:
                next_return = next_values[i].detach()
            returns[i] = rewards[i] + self.args.gamma * next_return
        deltas = returns - values
        advantages = values_pol
        if self.args.normalize_advantages:
            advantages = batchnorm(advantages)
        action_loss = - advantages
        # action_loss = action_loss.mean(dim=0)
        # value_loss = deltas.pow(2).mean(dim=0)
        action_loss = action_loss.mean()
        value_loss = deltas.pow(2).mean()
        return action_loss, value_loss, action_out
