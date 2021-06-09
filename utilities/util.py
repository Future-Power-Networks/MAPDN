import numpy as np
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.normal import Normal
from collections import namedtuple


class GumbelSoftmax(OneHotCategorical):
    def __init__(self, logits, probs=None, temperature=0.1):
        super(GumbelSoftmax, self).__init__(logits=logits, probs=probs)
        self.eps = 1e-20
        self.temperature = temperature

    def sample_gumbel(self):
        U = self.logits.clone()
        U.uniform_(0, 1)
        return -torch.log( -torch.log( U + self.eps ) )

    def gumbel_softmax_sample(self):
        y = self.logits + self.sample_gumbel()
        return torch.softmax( y / self.temperature, dim=-1)

    def hard_gumbel_softmax_sample(self):
        y = self.gumbel_softmax_sample()
        return (torch.max(y, dim=-1, keepdim=True)[0] == y).float()

    def rsample(self):
        return self.gumbel_softmax_sample()

    def sample(self):
        return self.rsample().detach()

    def hard_sample(self):
        return self.hard_gumbel_softmax_sample()

def normal_entropy(mean, std):
    return Normal(mean, std).entropy().sum()

def multinomial_entropy(logits):
    assert logits.size(-1) > 1
    return GumbelSoftmax(logits=logits).entropy().sum()

def normal_log_density(actions, means, log_stds):
    stds = log_stds.exp()
    return Normal(means, stds).log_prob(actions)

def multinomials_log_density(actions, logits):
    assert logits.size(-1) > 1
    return GumbelSoftmax(logits=logits).log_prob(actions)

def select_action(args, logits, status='train', exploration=True, info={}):
    if args.continuous:
        act_mean = logits
        act_std = info['log_std'].exp()
        if status == 'train':
            if exploration:
                if info.get('clip', False):
                    normal = Normal(torch.zeros_like(act_mean), act_std)
                    x_t = normal.rsample()
                    actions = act_mean + torch.clamp(x_t, min=-args.c, max=-args.c)
                    return actions, None
                elif info.get('enforcing_action_bound', False):
                    normal = Normal(act_mean, act_std)
                    x_t = normal.rsample()
                    y_t = torch.tanh(x_t)
                    actions = y_t * args.action_scale + args.action_bias
                    log_prob = normal.log_prob(x_t)
                    # Enforcing Action Bound
                    log_prob -= torch.log(args.action_scale * (1 - y_t.pow(2)) +  1e-6)
                    return actions, log_prob
                else:
                    normal = Normal(act_mean, act_std)
                    x_t = normal.rsample()
                    log_prob = normal.log_prob(x_t)
                    actions = x_t
                    return actions, log_prob
            else:
                return act_mean, None
        elif status == 'test':
            return act_mean, None
    else:
        if status == 'train':
            if exploration:
                if args.epsilon_softmax:
                    eps = info['softmax_eps']
                    p_a = (1 - eps) * torch.softmax(logits, dim=-1) + eps / logits.size(-1)
                    categorical = OneHotCategorical(logits=None, probs=p_a)
                    actions = categorical.sample()
                    log_prob = categorical.log_prob(actions)
                    return actions, log_prob
                elif args.gumbel_softmax:
                    gumbel = GumbelSoftmax(logits=logits)
                    actions = gumbel.rsample()
                    log_prob = gumbel.log_prob(actions)
                    return actions, log_prob
                else:
                    categorical = OneHotCategorical(logits=logits)
                    actions = categorical.sample()
                    log_prob = categorical.log_prob(actions)
                    return actions, log_prob
            else:
                if args.gumbel_softmax:
                    gumbel = GumbelSoftmax(logits=logits, temperature=1.0)
                    actions = gumbel.sample()
                    log_prob = gumbel.log_prob(actions)
                    return actions, log_prob
                else:
                    categorical = OneHotCategorical(logits=logits)
                    actions = categorical.sample()
                    log_prob = categorical.log_prob(actions)
                    return actions, log_prob
        elif status == 'test':
            p_a = torch.softmax(logits, dim=-1)
            return  (p_a == torch.max(p_a, dim=-1, keepdim=True)[0]).float(), None

def translate_action(args, action, env):
    if args.continuous:
        actions = action.detach().squeeze().cpu().numpy()
        cp_actions = actions.copy()
        # clip and scale action to correct range
        low = env.action_space.low
        high = env.action_space.high
        for i in range(len(cp_actions)):
            cp_actions[i] = max(-1.0, min(cp_actions[i], 1.0))
            cp_actions[i] = 0.5 * (cp_actions[i] + 1.0) * (high - low) + low
        return actions, cp_actions
    else:
        actual = [act.detach().squeeze().cpu().numpy() for act in torch.unbind(action, 1)]
        return action, actual

def prep_obs(state=[]):
    state = np.array(state)
    # for single transition -> batch_size=1
    if len(state.shape) == 2:
        state = np.stack(state, axis=0)
    # for single episode
    elif len(state.shape) == 4:
        state = np.concatenate(state, axis=0)
    else:
        raise RuntimeError('The shape of the observation is incorrect.')
    return torch.tensor(state).float()

def cuda_wrapper(tensor, cuda):
    if isinstance(tensor, torch.Tensor):
        return tensor.cuda() if cuda else tensor
    else:
        raise RuntimeError('Please enter a pytorch tensor, now a {} is received.'.format(type(tensor)))

def batchnorm(batch):
    if isinstance(batch, torch.Tensor):
        return (batch - batch.mean(dim=0)) / (batch.std(dim=0) + 1e-7)
    else:
        raise RuntimeError('Please enter a pytorch tensor, now a {} is received.'.format(type(batch)))

def get_grad_norm(params):
    grad_norms = []
    for param in params:
        grad_norms.append(torch.norm(param.grad).item())
    return np.mean(grad_norms)

def merge_dict(stat, key, value):
    if key in stat.keys():
        stat[key] += value
    else:
        stat[key] = value

def n_step(rewards, last_step, done, next_values, n_step, args):
    cuda = torch.cuda.is_available() and args.cuda
    returns = cuda_wrapper(torch.zeros_like(rewards), cuda=cuda)
    i = rewards.size(0)-1
    while i >= 0:
        if last_step[i]:
            next_return = 0 if done[i] else next_values[i].detach()
            for j in reversed(range(i-n_step+1, i+1)):
                returns[j] = rewards[j] + args.gamma * next_return
                next_return = returns[j]
            i -= n_step
            continue
        else:
            next_return = next_values[i+n_step-1].detach()
        for j in reversed(range(n_step)):
            g = rewards[i+j] + args.gamma * next_return
            next_return = g
        returns[i] = g.detach()
        i -= 1
    return returns

def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)