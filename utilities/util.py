import numpy as np
import torch as th
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
        return -th.log( -th.log( U + self.eps ) )

    def gumbel_softmax_sample(self):
        y = self.logits + self.sample_gumbel()
        return th.softmax( y / self.temperature, dim=-1)

    def hard_gumbel_softmax_sample(self):
        y = self.gumbel_softmax_sample()
        return (th.max(y, dim=-1, keepdim=True)[0] == y).float()

    def rsample(self):
        return self.gumbel_softmax_sample()

    def sample(self):
        return self.rsample().detach()

    def hard_sample(self):
        return self.hard_gumbel_softmax_sample()

def normal_entropy(mean, std):
    return Normal(mean, std).entropy().mean()

def multinomial_entropy(logits):
    assert logits.size(-1) > 1
    return GumbelSoftmax(logits=logits).entropy().mean()

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
                if args.action_enforcebound:
                    normal = Normal(act_mean, act_std)
                    x_t = normal.rsample()
                    y_t = th.tanh(x_t)
                    log_prob = normal.log_prob(x_t)
                    # Enforcing Action Bound
                    log_prob -= th.log(1 - y_t.pow(2) + 1e-6)
                    actions = y_t
                    return actions, log_prob
                else:
                    normal = Normal(th.zeros_like(act_mean), act_std)
                    x_t = normal.rsample()
                    log_prob = normal.log_prob(x_t)
                    # this is usually for target value
                    if info.get('clip', False):
                        actions = act_mean + th.clamp(x_t, min=-args.clip_c, max=args.clip_c)
                    else:
                        actions = act_mean + x_t
                    return actions, log_prob
            else:
                actions = act_mean
                return actions, None
        elif status == 'test':
            if args.action_enforcebound:
                x_t = act_mean
                actions = th.tanh(x_t)
                return actions, None
            else:
                actions = act_mean
                return actions, None
    else:
        if status == 'train':
            if exploration:
                if args.epsilon_softmax:
                    eps = args.softmax_eps
                    p_a = (1 - eps) * th.softmax(logits, dim=-1) + eps / logits.size(-1)
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
            p_a = th.softmax(logits, dim=-1)
            return  (p_a == th.max(p_a, dim=-1, keepdim=True)[0]).float(), None

def translate_action(args, action, env):
    if args.continuous:
        actions = action.detach().squeeze()
        # clip and scale action to correct range for safety
        cp_actions = th.clamp(actions, min=-1.0, max=1.0)
        low = args.action_bias - args.action_scale
        high = args.action_bias + args.action_scale
        cp_actions = 0.5 * (cp_actions + 1.0) * (high - low) + low
        cp_actions = cp_actions.cpu().numpy()
        return actions, cp_actions
    else:
        actual = [act.detach().squeeze().cpu().numpy() for act in th.unbind(action, 1)]
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
    return th.tensor(state).float()

def cuda_wrapper(tensor, cuda):
    if isinstance(tensor, th.Tensor):
        return tensor.cuda() if cuda else tensor
    else:
        raise RuntimeError('Please enter a pyth tensor, now a {} is received.'.format(type(tensor)))

def batchnorm(batch):
    if isinstance(batch, th.Tensor):
        return (batch - batch.mean(dim=0)) / (batch.std(dim=0) + 1e-7)
    else:
        raise RuntimeError('Please enter a pytorch tensor, now a {} is received.'.format(type(batch)))

def get_grad_norm(args, params):
    grad_norms = th.nn.utils.clip_grad_norm_(params, args.grad_clip_eps)
    return grad_norms

def merge_dict(stat, key, value):
    if key in stat.keys():
        stat[key] += value
    else:
        stat[key] = value

def n_step(rewards, last_step, done, next_values, n_step, args):
    cuda = th.cuda.is_available() and args.cuda
    returns = cuda_wrapper(th.zeros_like(rewards), cuda=cuda)
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

def dict2str(dict, dict_name):
    string = [f'{dict_name}:']
    for k, v in dict.items():
        string.append(f'\t{k}: {v}' )
    string = "\n".join(string)
    return string