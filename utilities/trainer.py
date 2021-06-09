from collections import namedtuple
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utilities.util import cuda_wrapper, multinomial_entropy, get_grad_norm, normal_entropy, batchnorm
from utilities.replay_buffer import TransReplayBuffer, EpisodeReplayBuffer


class PGTrainer(object):
    def __init__(self, args, model, env, logger):
        self.args = args
        self.cuda_ = self.args.cuda and torch.cuda.is_available()
        self.logger = logger
        self.episodic = self.args.episodic
        if self.args.target:
            target_net = model(self.args).cuda() if self.cuda_ else model(self.args)
            self.behaviour_net = model(self.args, target_net).cuda() if self.cuda_ else model(self.args, target_net)
        else:
            self.behaviour_net = model(self.args).cuda() if self.cuda_ else model(self.args)
        if self.args.replay:
            if not self.episodic:
                self.replay_buffer = TransReplayBuffer(int(self.args.replay_buffer_size))
            else:
                self.replay_buffer = EpisodeReplayBuffer(int(self.args.replay_buffer_size))
        self.env = env
        self.policy_optimizer = optim.RMSprop(self.behaviour_net.policy_dicts.parameters(), lr=args.policy_lrate)
        self.value_optimizer = optim.RMSprop(self.behaviour_net.value_dicts.parameters(), lr=args.value_lrate)
        self.init_action = cuda_wrapper( torch.zeros(1, self.args.agent_num, self.args.action_dim), cuda=self.cuda_ )
        self.steps = 0
        self.episodes = 0
        self.entr = self.args.entr

    def get_loss(self, batch):
        policy_loss, value_loss, logits = self.behaviour_net.get_loss(batch)
        return policy_loss, value_loss, logits

    def policy_compute_grad(self, stat, loss, retain_graph):
        if self.entr > 0:
            if self.args.continuous:
                policy_loss, means, log_stds = loss
                entropy = normal_entropy(means, log_stds.exp())
            else:
                policy_loss, logits = loss
                entropy = multinomial_entropy(logits)
            policy_loss -= self.entr * entropy
            stat['entropy'] = entropy.item()
        policy_loss.backward(retain_graph=retain_graph)

    def value_compute_grad(self, value_loss, retain_graph):
        value_loss.backward(retain_graph=retain_graph)

    def grad_clip(self, params):
        for param in params:
            param.grad.data.clamp_(-self.args.grad_clip_eps, self.args.grad_clip_eps)

    def policy_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        self.policy_transition_process(stat, batch)

    def value_replay_process(self, stat):
        batch = self.replay_buffer.get_batch(self.args.batch_size)
        batch = self.behaviour_net.Transition(*zip(*batch))
        self.value_transition_process(stat, batch)

    def policy_transition_process(self, stat, trans):
        if self.args.continuous:
            policy_loss, _, logits = self.get_loss(trans)
            means, log_stds = logits
        else:
            policy_loss, _, logits = self.get_loss(trans)
        self.policy_optimizer.zero_grad()
        if self.args.continuous:
            self.policy_compute_grad(stat, (policy_loss, means, log_stds), False)
        else:
            self.policy_compute_grad(stat, (policy_loss, logits), False)
        param = self.policy_optimizer.param_groups[0]['params']
        if self.args.grad_clip:
            self.grad_clip(param)
        policy_grad_norms = get_grad_norm(param)
        self.policy_optimizer.step()
        stat['policy_grad_norm'] = np.array(policy_grad_norms).mean()
        stat['policy_loss'] = policy_loss.clone().mean().item()

    def value_transition_process(self, stat, trans):
        _, value_loss, _ = self.get_loss(trans)
        self.value_optimizer.zero_grad()
        self.value_compute_grad(value_loss, False)
        param = self.value_optimizer.param_groups[0]['params']
        if self.args.grad_clip:
            self.grad_clip(param)
        value_grad_norms = get_grad_norm(param)
        self.value_optimizer.step()
        stat['value_grad_norm'] = np.array(value_grad_norms).mean()
        stat['value_loss'] = value_loss.mean().item()

    def run(self, stat, episode):
        self.behaviour_net.train_process(stat, self)
        if episode%self.args.eval_freq == 0:
            self.behaviour_net.evaluation(stat, self)

    def logging(self, stat):
        for k, v in stat.items():
            self.logger.add_scalar('data/'+k, v, self.episodes)

    def print_info(self, stat):
        string = [f'Episode: {self.episodes}']
        for k, v in stat.items():
            string.append(k+f': {v:2.4f}')
        string = "\n".join(string)
        print (string)
