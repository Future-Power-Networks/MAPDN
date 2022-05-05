import torch as th
import torch.nn as nn
import numpy as np
from collections import namedtuple
from utilities.util import prep_obs, translate_action



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.device = th.device( "cuda" if th.cuda.is_available() and self.args.cuda else "cpu" )
        self.n_ = self.args.agent_num
        self.hid_dim = self.args.hid_size
        self.obs_dim = self.args.obs_size
        self.act_dim = self.args.action_dim
        self.Transition = namedtuple('Transition', ('state', 'action', 'log_prob_a', 'value', 'next_value', 'reward', 'next_state', 'done', 'last_step', 'action_avail', 'last_hid', 'hid'))
        self.batchnorm = nn.BatchNorm1d(self.n_)
        
    def reload_params_to_target(self):
        self.target_net.policy_dicts.load_state_dict( self.policy_dicts.state_dict() )
        self.target_net.value_dicts.load_state_dict( self.value_dicts.state_dict() )
        if self.args.mixer:
            self.target_net.mixer.load_state_dict( self.mixer.state_dict() )

    def update_target(self):
        for name, param in self.target_net.policy_dicts.state_dict().items():
            update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.policy_dicts.state_dict()[name]
            self.target_net.policy_dicts.state_dict()[name].copy_(update_params)
        for name, param in self.target_net.value_dicts.state_dict().items():
            update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.value_dicts.state_dict()[name]
            self.target_net.value_dicts.state_dict()[name].copy_(update_params)
        if self.args.mixer:
            for name, param in self.target_net.mixer.state_dict().items():
                update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.mixer.state_dict()[name]
                self.target_net.mixer.state_dict()[name].copy_(update_params)

    def transition_update(self, trainer, trans, stat):
        if self.args.replay:
            trainer.replay_buffer.add_experience(trans)
            replay_cond = trainer.steps>self.args.replay_warmup\
             and len(trainer.replay_buffer.buffer)>=self.args.batch_size\
             and trainer.steps%self.args.behaviour_update_freq==0
            if replay_cond:
                for _ in range(self.args.value_update_epochs):
                    trainer.value_replay_process(stat)
                for _ in range(self.args.policy_update_epochs):
                    trainer.policy_replay_process(stat)
                if self.args.mixer:
                    for _ in range(self.args.mixer_update_epochs):
                        trainer.mixer_replay_process(stat)
                # TODO: hard code
                # clear replay buffer for on-policy algorithm
                if self.__class__.__name__ in ["COMA", "IAC", "IPPO", "MAPPO"] :
                    trainer.replay_buffer.clear()
        else:
            trans_cond = trainer.steps%self.args.behaviour_update_freq==0
            if trans_cond:
                for _ in range(self.args.value_update_epochs):
                    trainer.value_replay_process(stat)
                for _ in range(self.args.policy_update_epochs):
                    trainer.policy_replay_process(stat, trans)
                if self.args.mixer:
                    for _ in range(self.args.mixer_update_epochs):
                        trainer.mixer_replay_process(stat)
        if self.args.target:
            target_cond = trainer.steps%self.args.target_update_freq==0
            if target_cond:
                self.update_target()

    def episode_update(self, trainer, episode, stat):
        if self.args.replay:
            trainer.replay_buffer.add_experience(episode)
            replay_cond = trainer.episodes>self.args.replay_warmup\
             and len(trainer.replay_buffer.buffer)>=self.args.batch_size\
             and trainer.episodes%self.args.behaviour_update_freq==0
            if replay_cond:
                for _ in range(self.args.value_update_epochs):
                    trainer.value_replay_process(stat)
                for _ in range(self.args.policy_update_epochs):
                    trainer.policy_replay_process(stat)
                if self.args.mixer:
                    for _ in range(self.args.mixer_update_epochs):
                        trainer.mixer_replay_process(stat)
        else:
            episode = self.Transition(*zip(*episode))
            episode_cond = trainer.episodes%self.args.behaviour_update_freq==0
            if episode_cond:
                for _ in range(self.args.value_update_epochs):
                    trainer.value_replay_process(stat)
                for _ in range(self.args.policy_update_epochs):
                    trainer.policy_replay_process(stat)
                if self.args.mixer:
                    for _ in range(self.args.mixer_update_epochs):
                        trainer.mixer_replay_process(stat)

    def construct_model(self):
        raise NotImplementedError()

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        # obs_shape = (b, n, o)
        batch_size = obs.size(0)

        # add agent id
        if self.args.agent_id:
            agent_ids = th.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device) # shape = (b, n, n)
            obs = th.cat( (obs, agent_ids), dim=-1 ) # shape = (b, n, n+o)

        if self.args.shared_params:
            # print (f"This is the shape of last_hids: {last_hid.size()}")
            obs = obs.contiguous().view(batch_size*self.n_, -1) # shape = (b*n, n+o/o)
            agent_policy = self.policy_dicts[0]
            means, log_stds, hiddens = agent_policy(obs, last_hid)
            # hiddens = th.stack(hiddens, dim=1)
            means = means.contiguous().view(batch_size, self.n_, -1)
            hiddens = hiddens.contiguous().view(batch_size, self.n_, -1)
            if self.args.gaussian_policy:
                log_stds = log_stds.contiguous().view(batch_size, self.n_, -1)
            else:
                stds = th.ones_like(means).to(self.device) * self.args.fixed_policy_std
                log_stds = th.log(stds)
        else:
            means = []
            hiddens = []
            log_stds = []
            for i, agent_policy in enumerate(self.policy_dicts):
                mean, log_std, hidden = agent_policy(obs[:, i, :], last_hid[:, i, :])
                means.append(mean)
                hiddens.append(hidden)
                log_stds.append(log_std)
            means = th.stack(means, dim=1)
            hiddens = th.stack(hiddens, dim=1)
            if self.args.gaussian_policy:
                log_stds = th.stack(log_stds, dim=1)
            else:
                log_stds = th.zeros_like(means).to(self.device)

        return means, log_stds, hiddens

    def value(self, obs, act, last_act=None, last_hid=None):
        raise NotImplementedError()

    def construct_policy_net(self):
        if self.args.agent_id:
            input_shape = self.obs_dim + self.n_
        else:
            input_shape = self.obs_dim

        if self.args.agent_type == 'mlp':
            if self.args.gaussian_policy:
                from agents.mlp_agent_gaussian import MLPAgent
            else:
                from agents.mlp_agent import MLPAgent
            Agent = MLPAgent
        elif self.args.agent_type == 'rnn':
            if self.args.gaussian_policy:
                from agents.rnn_agent_gaussian import RNNAgent
            else:
                from agents.rnn_agent import RNNAgent
            Agent = RNNAgent
        else:
            NotImplementedError()
            
        if self.args.shared_params:
            self.policy_dicts = nn.ModuleList([ Agent(input_shape, self.args) ])
        else:
            self.policy_dicts = nn.ModuleList([ Agent(input_shape, self.args) for _ in range(self.n_) ])

    def construct_value_net(self):
        raise NotImplementedError()

    def init_weights(self, m):
        '''
        initialize the weights of parameters
        '''
        if type(m) == nn.Linear:
            if self.args.init_type == "normal":
                nn.init.normal_(m.weight, 0.0, self.args.init_std)
            elif self.args.init_type == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain(self.args.hid_activation))

    def get_actions(self):
        raise NotImplementedError()

    def get_loss(self):
        raise NotImplementedError()

    def credit_assignment_demo(self, obs, act):
        assert isinstance(obs, np.ndarray)
        assert isinstance(act, np.ndarray)
        obs = th.tensor(obs).to(self.device).float()
        act = th.tensor(act).to(self.device).float()
        values = self.value(obs, act)
        return values

    def train_process(self, stat, trainer):
        stat_train = {'mean_train_reward': 0}

        if self.args.episodic:
            episode = []

        # reset env
        state, global_state = trainer.env.reset()

        # init hidden states
        last_hid = self.policy_dicts[0].init_hidden()

        for t in range(self.args.max_steps):
            # current state, action, value
            state_ = prep_obs(state).to(self.device).contiguous().view(1, self.n_, self.obs_dim)
            action, action_pol, log_prob_a, _, hid = self.get_actions(state_, status='train', exploration=True, actions_avail=th.tensor(trainer.env.get_avail_actions()), target=False, last_hid=last_hid)
            value = self.value(state_, action_pol)
            _, actual = translate_action(self.args, action, trainer.env)
            # reward
            reward, done, info = trainer.env.step(actual)
            reward_repeat = [reward]*trainer.env.get_num_of_agents()
            # next state, action, value
            next_state = trainer.env.get_obs()
            next_state_ = prep_obs(next_state).to(self.device).contiguous().view(1, self.n_, self.obs_dim)
            _, next_action_pol, _, _, _ = self.get_actions(next_state_, status='train', exploration=True, actions_avail=th.tensor(trainer.env.get_avail_actions()), target=False, last_hid=hid)
            next_value = self.value(next_state_, next_action_pol)
            # store trajectory
            if isinstance(done, list): done = np.sum(done)
            done_ = done or t==self.args.max_steps-1
            trans = self.Transition(state,
                                    action_pol.detach().cpu().numpy(),
                                    log_prob_a.detach().cpu().numpy(),
                                    value.detach().cpu().numpy(),
                                    next_value.detach().cpu().numpy(),
                                    np.array(reward_repeat),
                                    next_state,
                                    done,
                                    done_,
                                    trainer.env.get_avail_actions(),
                                    last_hid.detach().cpu().numpy(),
                                    hid.detach().cpu().numpy()
                                   )
            if not self.args.episodic:
                self.transition_update(trainer, trans, stat)
            else:
                episode.append(trans)
            for k, v in info.items():
                if 'mean_train_'+k not in stat_train.keys():
                    stat_train['mean_train_' + k] = v
                else:
                    stat_train['mean_train_' + k] += v
            stat_train['mean_train_reward'] += reward
            trainer.steps += 1
            if done_:
                break
            # set the next state
            state = next_state
            # set the next last_hid
            last_hid = hid
        trainer.episodes += 1
        for k, v in stat_train.items():
            key_name = k.split('_')
            if key_name[0] == 'mean':
                stat_train[k] = v / float(t+1)
        stat.update(stat_train)
        if self.args.episodic:
            self.episode_update(trainer, episode, stat)

    def evaluation(self, stat, trainer):
        num_eval_episodes = self.args.num_eval_episodes
        stat_test = {}
        for _ in range(num_eval_episodes):
            stat_test_epi = {'mean_test_reward': 0}
            state, global_state = trainer.env.reset()
            # init hidden states
            last_hid = self.policy_dicts[0].init_hidden()
            for t in range(self.args.max_steps):
                state_ = prep_obs(state).to(self.device).contiguous().view(1, self.n_, self.obs_dim)
                action, _, _, _, hid = self.get_actions(state_, status='test', exploration=False, actions_avail=th.tensor(trainer.env.get_avail_actions()), target=False, last_hid=last_hid)
                _, actual = translate_action(self.args, action, trainer.env)
                reward, done, info = trainer.env.step(actual)
                done_ = done or t==self.args.max_steps-1
                next_state = trainer.env.get_obs()
                if isinstance(done, list): done = np.sum(done)
                for k, v in info.items():
                    if 'mean_test_' + k not in stat_test_epi.keys():
                        stat_test_epi['mean_test_' + k] = v
                    else:
                        stat_test_epi['mean_test_' + k] += v
                stat_test_epi['mean_test_reward'] += reward
                if done_:
                    break
                # set the next state
                state = next_state
                # set the next last_hid
                last_hid = hid
            for k, v in stat_test_epi.items():
                stat_test_epi[k] = v / float(t+1)
            for k, v in stat_test_epi.items():
                if k not in stat_test.keys():
                    stat_test[k] = v
                else:
                    stat_test[k] += v
        for k, v in stat_test.items():
            stat_test[k] = v / float(num_eval_episodes)
        stat.update(stat_test)

    def unpack_data(self, batch):
        reward = th.tensor(batch.reward, dtype=th.float).to(self.device)
        last_step = th.tensor(batch.last_step, dtype=th.float).to(self.device).contiguous().view(-1, 1)
        done = th.tensor(batch.done, dtype=th.float).to(self.device).contiguous().view(-1, 1)
        action = th.tensor(np.concatenate(batch.action, axis=0), dtype=th.float).to(self.device)
        log_prob_a = th.tensor(np.concatenate(batch.action, axis=0), dtype=th.float).to(self.device)
        value = th.tensor(np.concatenate(batch.value, axis=0), dtype=th.float).to(self.device)
        next_value = th.tensor(np.concatenate(batch.next_value, axis=0), dtype=th.float).to(self.device)
        state = prep_obs( list( zip(batch.state) ) ).to(self.device)
        next_state = prep_obs( list( zip(batch.next_state) ) ).to(self.device)
        action_avail = th.tensor( np.concatenate(batch.action_avail, axis=0) ).to(self.device)
        last_hid = th.tensor(np.concatenate(batch.last_hid, axis=0), dtype=th.float).to(self.device)
        hid = th.tensor(np.concatenate(batch.hid, axis=0), dtype=th.float).to(self.device)
        if self.args.reward_normalisation:
            reward = self.batchnorm(reward).to(self.device)
        return (state, action, log_prob_a, value, next_value, reward, next_state, done, last_step, action_avail, last_hid, hid)
