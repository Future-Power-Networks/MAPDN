from ..multiagentenv import MultiAgentEnv
import numpy as np
import pandapower as pp
from pandapower import ppException
import pandas as pd
import copy
import os
from collections import namedtuple


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


class ActionSpace(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high


class VoltageControl(MultiAgentEnv):
    def __init__(self, kwargs):
        '''initialisation
        '''
        # unpack args
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        # set the data path
        self.data_path = args.data_path

        # load the model of power network
        self.base_powergrid = self._load_network()
        
        # load data
        self.pv_data = self._load_pv_data()
        self.active_demand_data = self._load_active_demand_data()
        self.reactive_demand_data = self._load_reactive_demand_data()

        # define episode and rewards
        self.episode_limit = args.episode_limit
        self.reward_type = getattr(args, "reward_type", "sensitive")
        self.voltage_weight = getattr(args, "voltage_weight", 0.5)
        self.q_weight = getattr(args, "q_weight", 0.1)
        self.line_weight = getattr(args, "line_weight", 0.1)
        self.dv_dq_weight = getattr(args, "dq_dv_weight", 0.001)

        # define constraints and uncertainty
        self.v_upper = getattr(args, "v_upper", 1.05)
        self.v_lower = getattr(args, "v_lower", 0.95)
        self.active_demand_std = self.active_demand_data.values.std(axis=0) / 100.0
        self.reactive_demand_std = self.reactive_demand_data.values.std(axis=0) / 100.0
        self.pv_std = self.pv_data.values.std(axis=0) / 100.0
        self._set_reactive_power_boundry()

        # define action space and observation space
        self.action_space = ActionSpace(low=-self.args.action_scale+self.args.action_bias, high=self.args.action_scale+self.args.action_bias)
        self.forecast_horizon = getattr(args, "forecast_horizon", 1)
        self.state_space = getattr(args, "state_space", ["pv", "demand", "reactive", "vm_pu", "va_degree"])
        if self.args.mode == "distributed":
            self.n_actions = 1
            self.n_agents = len(self.base_powergrid.sgen)
        elif self.args.mode == "decentralised":
            self.n_actions = len(self.base_powergrid.sgen)
            self.n_agents = len( set( self.base_powergrid.bus["zone"].to_numpy(copy=True) ) ) - 1 # exclude the main zone
        agents_obs, state = self.reset()

        self.obs_size = agents_obs[0].shape[0]
        # self.obs_size = agents_obs[0].shape[1]
        self.state_size = state.shape[0]
        self.last_v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        self.last_q = self.powergrid.sgen["q_mvar"].to_numpy(copy=True)

    def reset(self, reset_time=True):
        """reset the env
        """
        # reset the time step, cumulative rewards and obs history
        self.steps = 1
        self.sum_rewards = 0
        if self.forecast_horizon > 1:
            self.obs_history = {i: [] for i in range(self.n_agents)}
        # reset the power grid
        self.powergrid = copy.deepcopy(self.base_powergrid)
        solvable = False
        while not solvable:
            # reset the time stamp
            if reset_time:
                self._episode_start_hour = self._select_start_hour()
                self._episode_start_day = self._select_start_day()
                self._episode_start_quarter = self._select_start_quarter()
            # get one episode of data
            self.pv_forecasts = self._get_episode_pv_forecast()
            self.active_demand_forecasts = self._get_episode_active_demand_forecast()
            self.reactive_demand_forecasts = self._get_episode_reactive_demand_forecast()
            self._set_demand_and_pv()
            # random initialise action
            if self.args.reset_action:
                self.powergrid.sgen["q_mvar"] = self.get_action()
                self.powergrid.sgen["q_mvar"] = self._clip_reactive_power(self.powergrid.sgen["q_mvar"], self.powergrid.sgen["p_mw"])
            try:    
                pp.runpp(self.powergrid)
                solvable = True
            except ppException:
                print ("The power flow in the demand and PV renew cannot be solved.")
                print (f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
                print (f"This is the q: \n{self.powergrid.sgen['q_mvar']}")
                print (f"This is the active demand: \n{self.powergrid.load['p_mw']}")
                print (f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
                print (f"This is the res_bus: \n{self.powergrid.res_bus}")
                solvable = False
        return self.get_obs(), self.get_state()
    
    def manual_reset(self, day, hour, quarter):
        # reset the time step, cumulative rewards and obs history
        self.steps = 1
        self.sum_rewards = 0
        if self.forecast_horizon > 1:
            self.obs_history = {i: [] for i in range(self.n_agents)}
        # reset the power grid
        self.powergrid = copy.deepcopy(self.base_powergrid)
        # reset the time stamp
        self._episode_start_hour = hour
        self._episode_start_day = day
        self._episode_start_quarter = quarter
        solvable = False
        while not solvable:
            # get one episode of data
            self.pv_forecasts = self._get_episode_pv_forecast()
            self.active_demand_forecasts = self._get_episode_active_demand_forecast()
            self.reactive_demand_forecasts = self._get_episode_reactive_demand_forecast()
            self._set_demand_and_pv(add_noise=False)
            # random initialise action
            if self.args.reset_action:
                self.powergrid.sgen["q_mvar"] = self.get_action()
                self.powergrid.sgen["q_mvar"] = self._clip_reactive_power(self.powergrid.sgen["q_mvar"], self.powergrid.sgen["p_mw"])
            try:    
                pp.runpp(self.powergrid)
                solvable = True
            except ppException:
                print ("The power flow in the demand and PV renew cannot be solved.")
                print (f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
                print (f"This is the q: \n{self.powergrid.sgen['q_mvar']}")
                print (f"This is the active demand: \n{self.powergrid.load['p_mw']}")
                print (f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
                print (f"This is the res_bus: \n{self.powergrid.res_bus}")
                solvable = False
        return self.get_obs(), self.get_state()

    def step(self, actions, add_noise=True):
        """function for the interaction between agent and the env each time step
        """
        last_powergrid = copy.deepcopy(self.powergrid)
        # check whether the power balance is unsolvable
        solvable = self._take_action(actions)
        if solvable:
            # get the reward of current actions
            reward, info = self._calc_reward()
        else:
            q_loss = np.mean( np.abs(self.powergrid.sgen["q_mvar"]) )
            self.powergrid = last_powergrid
            reward, info = self._calc_reward()
            reward -= 200.
            # keep q_loss
            info["destroy"] = 1.
            info["totally_controllable_ratio"] = 0.
            info["q_loss"] = q_loss
        # set the pv and demand for the next time step
        self._set_demand_and_pv(add_noise=add_noise)
        # terminate if episode_limit is reached
        self.steps += 1
        self.sum_rewards += reward
        if self.steps >= self.episode_limit or not solvable:
            terminated = True
        else:
            terminated = False
        if terminated:
            print (f"Episode terminated at time: {self.steps} with return: {self.sum_rewards:2.4f}.")
        return reward, terminated, info

    def get_state(self):
        """return the global state for the power system
        the default state: voltage
        auxiliary state: active power of generators, bus state, load active power, load reactive power
        """
        state = []
        if "demand" in self.state_space:
            # TODO: check the correctness
            state += list(self.powergrid.res_bus["p_mw"].sort_index().to_numpy(copy=True))
            state += list(self.powergrid.res_bus["q_mvar"].sort_index().to_numpy(copy=True))
        if "pv" in self.state_space:
            # TODO: check the correctness
            state += list(self.powergrid.sgen["p_mw"].sort_index().to_numpy(copy=True))
        if "reactive" in self.state_space:
            # TODO: check the correctness
            state += list(self.powergrid.sgen["q_mvar"].sort_index().to_numpy(copy=True))
        if "vm_pu" in self.state_space:
            # TODO: check the correctness
            state += list(self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True))
        if "va_degree" in self.state_space:
            # TODO: check the correctness
            state += list(self.powergrid.res_bus["va_degree"].sort_index().to_numpy(copy=True))
        state = np.array(state)
        return state
    
    def get_obs(self):
        """return the obs for each agent in the power system
        the default obs: voltage
        auxiliary obs: active power of generators, bus state, load active power, load reactive power
        each agent can only observe the state within its zone for both distributed and decentralised mode
        """
        clusters = self._get_clusters_info()
        if self.args.mode == "distributed":
            obs_zone_dict = dict()
            zone_list = list()
            obs_len_list = list()
            for i in range(len(self.powergrid.sgen)):
                obs = list()
                zone_buses, zone, pv, q, sgen_bus = clusters[f"sgen{i}"]
                zone_list.append(zone)
                if not( zone in obs_zone_dict.keys() ):
                    if "demand" in self.state_space:
                        copy_zone_buses = copy.deepcopy(zone_buses)
                        copy_zone_buses.loc[sgen_bus]["p_mw"] -= pv
                        copy_zone_buses.loc[sgen_bus]["q_mvar"] -= q
                        obs += list(copy_zone_buses.loc[:, "p_mw"].to_numpy(copy=True))
                        obs += list(copy_zone_buses.loc[:, "q_mvar"].to_numpy(copy=True))
                    if "pv" in self.state_space:
                        obs.append(pv)
                    if "reactive" in self.state_space:
                        obs.append(q)
                    if "vm_pu" in self.state_space:
                        obs += list(zone_buses.loc[:, "vm_pu"].to_numpy(copy=True))
                    if "va_degree" in self.state_space:
                        # transform the voltage phase to radius
                        obs += list(zone_buses.loc[:, "va_degree"].to_numpy(copy=True) * np.pi / 180)
                    obs_zone_dict[zone] = np.array(obs)
                obs_len_list.append(obs_zone_dict[zone].shape[0])
            agents_obs = list()
            obs_max_len = max(obs_len_list)
            for zone in zone_list:
                obs_zone = obs_zone_dict[zone]
                pad_obs_zone = np.concatenate( [obs_zone, np.zeros(obs_max_len - obs_zone.shape[0])], axis=0 )
                agents_obs.append(pad_obs_zone)
        elif self.args.mode == "decentralised":
            obs_len_list = list()
            zone_obs_list = list()
            for i in range(self.n_agents):
                zone_buses, pv, q, sgen_buses = clusters[f"zone{i+1}"]
                obs = list()
                if "demand" in self.state_space:
                    copy_zone_buses = copy.deepcopy(zone_buses)
                    copy_zone_buses.loc[sgen_buses]["p_mw"] -= pv
                    copy_zone_buses.loc[sgen_buses]["q_mvar"] -= q
                    obs += list(copy_zone_buses.loc[:, "p_mw"].to_numpy(copy=True))
                    obs += list(copy_zone_buses.loc[:, "q_mvar"].to_numpy(copy=True))
                if "pv" in self.state_space:
                    obs += list(pv.to_numpy(copy=True))
                if "reactive" in self.state_space:
                    obs += list(q.to_numpy(copy=True))
                if "vm_pu" in self.state_space:
                    obs += list(zone_buses.loc[:, "vm_pu"].to_numpy(copy=True))
                if "va_degree" in self.state_space:
                    obs += list(zone_buses.loc[:, "va_degree"].to_numpy(copy=True) * np.pi / 180)
                obs = np.array(obs)
                zone_obs_list.append(obs)
                obs_len_list.append(obs.shape[0])
            agents_obs = []
            obs_max_len = max(obs_len_list)
            for obs_zone in zone_obs_list:
                pad_obs_zone = np.concatenate( [obs_zone, np.zeros(obs_max_len - obs_zone.shape[0])], axis=0 )
                agents_obs.append(pad_obs_zone)
                # extend to the axis 0
                # agents_obs.append(pad_obs_zone[np.newaxis, :])
        if self.forecast_horizon > 1:
            agents_obs_ = []
            # obs_shape = (obs_shape,)
            for i, obs in enumerate(agents_obs):
                if len(self.obs_history[i]) >= self.forecast_horizon - 1:
                    obs_ = np.concatenate(self.obs_history[i][-self.forecast_horizon+1:]+[obs], axis=0)
                else:
                    zeros = [np.zeros_like(obs)] * ( self.forecast_horizon - len(self.obs_history[i]) - 1 )
                    obs_ = self.obs_history[i] + [obs]
                    obs_ = zeros + obs_
                    obs_ = np.concatenate(obs_, axis=0)
                agents_obs_.append(copy.deepcopy(obs_))
                self.obs_history[i].append(copy.deepcopy(obs))
            agents_obs = agents_obs_
        # label agent id
        # agents_obs_id = []
        # agents_ids = np.eye(self.n_agents)
        # for i, obs in enumerate(agents_obs):
        #     obs_id = np.concatenate( (obs, agents_ids[i]), axis=0 )
        #     agents_obs_id.append( copy.deepcopy(obs_id) )
        # agents_obs = agents_obs_id
        return agents_obs

    def get_obs_agent(self, agent_id):
        """return observation for agent_id 
        """
        agents_obs = self.get_obs()
        return agents_obs[agent_id]
    
    def get_obs_size(self):
        """return the observation size
        """
        return self.obs_size

    def get_state_size(self):
        """return the state size
        """
        return self.state_size

    def get_action(self):
        """return the action according to a uniform distribution over [action_lower, action_upper)
        """
        rand_action = np.random.uniform(low=self.action_space.low, high=self.action_space.high, size=self.powergrid.sgen["q_mvar"].values.shape)
        return rand_action

    def get_total_actions(self):
        """return the total number of actions an agent could ever take 
        """
        return self.n_actions

    def get_avail_actions(self):
        """return available actions for all agents
        """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return np.expand_dims(np.array(avail_actions), axis=0)

    def get_avail_agent_actions(self, agent_id):
        """ return the available actions for agent_id 
        """
        if self.args.mode == "distributed":
            return [1]
        elif self.args.mode == "decentralised":
            avail_actions = np.zeros(self.n_actions)
            zone_sgens = self.base_powergrid.sgen.loc[self.base_powergrid.sgen["name"] == f"zone{agent_id+1}"]
            avail_actions[zone_sgens.index] = 1
            return avail_actions

    def get_num_of_agents(self):
        """return the number of agents
        """
        return self.n_agents

    def _get_voltage(self):
        return self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)

    def _create_basenet(self, base_net):
        """initilization of power grid
        set the pandapower net to use
        """
        if base_net is None:
            raise Exception("Please provide a base_net configured as pandapower format.")
        else:
            return base_net

    def _select_start_hour(self):
        """select start hour for an episode
        """
        return np.random.choice(24)
    
    def _select_start_quarter(self):
        """select start quarter for an episode
        """
        return np.random.choice(4)

    def _select_start_day(self):
        """select start day (date) for an episode
        """
        pv_data = self.pv_data
        pv_days = (pv_data.index[-1] - pv_data.index[0])
        pv_days = pv_days.days
        episode_days = ( self.episode_limit // (24 * 4) ) + 1  # margin
        return np.random.choice(pv_days - episode_days)

    def _load_network(self):
        """load network
        """
        network_path = os.path.join(self.data_path, 'model.p')
        base_net = pp.from_pickle(network_path)
        return self._create_basenet(base_net)

    def _load_pv_data(self):
        """load pv data
        the sensor frequency is set to 15 mins as default
        """
        pv_path = os.path.join(self.data_path, 'pv_active.csv')
        pv = pd.read_csv(pv_path, index_col=None)
        pv.index = pd.to_datetime(pv.iloc[:, 0])
        pv.index.name = 'time'
        pv = pv.iloc[::1, 1:] * self.args.pv_scale
        return pv

    def _load_active_demand_data(self):
        """load active demand data
        the sensor frequency is set to 15 mins as default
        """
        demand_path = os.path.join(self.data_path, 'load_active.csv')
        demand = pd.read_csv(demand_path, index_col=None)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = 'time'
        demand = demand.iloc[::1, 1:] * self.args.demand_scale
        return demand
    
    def _load_reactive_demand_data(self):
        """load reactive demand data
        the sensor frequency is set to 15 mins as default
        """
        demand_path = os.path.join(self.data_path, 'load_reactive.csv')
        demand = pd.read_csv(demand_path, index_col=None)
        demand.index = pd.to_datetime(demand.iloc[:, 0])
        demand.index.name = 'time'
        demand = demand.iloc[::1, 1:] * self.args.demand_scale
        return demand

    def _get_episode_pv_forecast(self):
        """return the pv forecast in an episode
        """
        episode_length = self.episode_limit
        horizon = self.forecast_horizon
        # convert the start date to quarters
        start = self._episode_start_quarter + self._episode_start_hour * 4 + self._episode_start_day * 24 * 4
        nr_quarters = episode_length + horizon + 1  # margin of 1
        episode_pv_forecast = self.pv_data[start:start + nr_quarters].values
        return episode_pv_forecast
    
    def _get_episode_active_demand_forecast(self):
        """return the active power forecasts for all loads in an episode
        """
        episode_length = self.episode_limit
        horizon = self.forecast_horizon
        start = self._episode_start_quarter + self._episode_start_hour * 4 + self._episode_start_day * 24 * 4
        nr_quarters = episode_length + horizon + 1  # margin of 1
        episode_demand_forecast = self.active_demand_data[start:start + nr_quarters].values
        return episode_demand_forecast
    
    def _get_episode_reactive_demand_forecast(self):
        """return the reactive power forecasts for all loads in an episode
        """
        episode_length = self.episode_limit
        horizon = self.forecast_horizon
        start = self._episode_start_quarter + self._episode_start_hour * 4 + self._episode_start_day * 24 * 4
        nr_quarters = episode_length + horizon + 1  # margin of 1
        episode_demand_forecast = self.reactive_demand_data[start:start + nr_quarters].values
        return episode_demand_forecast

    def _get_pv_forecast(self):
        """returns pv forecast for the next lookahead hours
        """
        t = self.steps
        horizon = self.forecast_horizon
        return self.pv_forecasts[t:t+horizon, :]

    def _get_active_demand_forecast(self):
        """return the forecasted hourly demand for the next lookahead hours
        """
        t = self.steps
        horizon = self.forecast_horizon
        return self.active_demand_forecasts[t:t+horizon, :]
    
    def _get_reactive_demand_forecast(self):
        """return the forecasted hourly demand for the next lookahead hours
        """
        t = self.steps
        horizon = self.forecast_horizon
        return self.reactive_demand_forecasts[t:t+horizon, :]
    
    def _get_pv_forecast(self):
        """returns solar forecast for the next lookahead hours
        """
        t = self.steps
        horizon = self.forecast_horizon
        return self.pv_forecasts[t:t+horizon, :]

    # TODO: DOUBLE CHECK
    def _set_demand_and_pv(self, add_noise=True):
        """update the demand and pv production according to the forecasts with some i.i.d. noise.
        """ 
        pv = copy.copy(self._get_pv_forecast()[0, :])
        # add uncertainty to pv data with unit truncated gaussian (only positive accepted)
        if add_noise:
            pv += self.pv_std * np.abs(np.random.randn(*pv.shape))
        active_demand = copy.copy(self._get_active_demand_forecast()[0, :])
        # add uncertainty to active power of demand data with unit truncated gaussian (only positive accepted)
        if add_noise:
            active_demand += self.active_demand_std * np.abs(np.random.randn(*active_demand.shape))
        reactive_demand = copy.copy(self._get_reactive_demand_forecast()[0, :])
        # add uncertainty to reactive power of demand data with unit truncated gaussian (only positive accepted)
        if add_noise:
            reactive_demand += self.reactive_demand_std * np.abs(np.random.randn(*reactive_demand.shape))
        # update the record in the pandapower
        self.powergrid.sgen["p_mw"] = pv
        self.powergrid.load["p_mw"] = active_demand
        self.powergrid.load["q_mvar"] = reactive_demand

    def _set_reactive_power_boundry(self):
        """set the boundary of reactive power
        """
        self.factor = 1.2
        self.p_max = self.pv_data.to_numpy(copy=True).max(axis=0)
        self.s_max = self.factor * self.p_max
        print (f"This is the s_max: \n{self.s_max}")

    def _get_clusters_info(self):
        """return the clusters of info
        the clusters info is divided by predefined zone
        distributed: each zone is equipped with several PV generators and each PV generator is an agent
        decentralised: each zone is controlled by an agent and each agent may have variant number of actions
        """
        clusters = dict()
        if self.args.mode == "distributed":
            for i in range(len(self.powergrid.sgen)):
                zone = self.powergrid.sgen["name"][i]
                sgen_bus = self.powergrid.sgen["bus"][i]
                pv = self.powergrid.sgen["p_mw"][i]
                q = self.powergrid.sgen["q_mvar"][i]
                zone_res_buses = self.powergrid.res_bus.sort_index().loc[self.powergrid.bus["zone"]==zone]
                clusters[f"sgen{i}"] = (zone_res_buses, zone, pv, q, sgen_bus)
        elif self.args.mode == "decentralised":
            for i in range(self.n_agents):
                zone_res_buses = self.powergrid.res_bus.sort_index().loc[self.powergrid.bus["zone"]==f"zone{i+1}"]
                sgen_res_buses = self.powergrid.sgen["bus"].loc[self.powergrid.sgen["name"] == f"zone{i+1}"]
                pv = self.powergrid.sgen["p_mw"].loc[self.powergrid.sgen["name"] == f"zone{i+1}"]
                q = self.powergrid.sgen["q_mvar"].loc[self.powergrid.sgen["name"] == f"zone{i+1}"]
                clusters[f"zone{i+1}"] = (zone_res_buses, pv, q, sgen_res_buses)
        return clusters
    
    def _take_action(self, actions):
        """take the control variables
        the control variables we consider are the exact reactive power
        of each distributed generator
        """
        self.powergrid.sgen["q_mvar"] = self._clip_reactive_power(actions, self.powergrid.sgen["p_mw"])
        # solve power flow to get the latest voltage with new reactive power and old deamnd and PV active power
        try:
            pp.runpp(self.powergrid)
            return True
        except ppException:
            print ("The power flow in the reactive power insert cannot be solved.")
            print (f"This is the pv: \n{self.powergrid.sgen['p_mw']}")
            print (f"This is the q: \n{self.powergrid.sgen['q_mvar']}")
            print (f"This is the active demand: \n{self.powergrid.load['p_mw']}")
            print (f"This is the reactive demand: \n{self.powergrid.load['q_mvar']}")
            print (f"This is the res_bus: \n{self.powergrid.res_bus}")
            return False
    
    def _clip_reactive_power(self, reactive_actions, active_power):
        """clip the reactive power to the hard safety range
        """
        reactive_power_constraint = np.sqrt(self.s_max**2 - active_power**2)
        return reactive_power_constraint * reactive_actions
    
    def _calc_reward(self, info={}):
        """reward function
        consider 4 possible choices on voltage loss:
            l1 loss
            l2 loss
            liu loss
            bowl loss
        """
        # percentage of voltage out of control
        v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        percent_of_v_out_of_control = ( np.sum(v < self.v_lower) + np.sum(v > self.v_upper) ) / v.shape[0]
        info["percentage_of_v_out_of_control"] = percent_of_v_out_of_control
        info["percentage_of_lower_than_lower_v"] = np.sum(v < self.v_lower) / v.shape[0]
        info["percentage_of_higher_than_upper_v"] = np.sum(v > self.v_upper) / v.shape[0]
        info["totally_controllable_ratio"] = 0. if percent_of_v_out_of_control > 1e-3 else 1.
        # v_main = self.powergrid.res_bus['vm_pu'].loc[self.powergrid.bus['zone']=='main']
        # info["percentage_of_v_out_of_control_no_main"] = percent_of_v_out_of_control - ( np.sum(v_main < self.v_lower) + np.sum(v_main > self.v_upper) ) / v.shape[0]
        # info["totally_controllable_ratio_no_main"] = 0. if info["percentage_of_v_out_of_control_no_main"] > 1e-3 else 1.
        # voltage violation
        v_ref = 0.5 * (self.v_lower + self.v_upper)
        info["average_voltage_deviation"] = np.mean( np.abs( v - v_ref ) )
        info["average_voltage"] = np.mean(v)
        # info["max_voltage_drop_deviation"] = np.max( (v < v_ref) * (v_ref - v) )
        # info["max_voltage_rise_deviation"] = np.max( (v > v_ref) * (v - v_ref) )
        info["max_voltage_drop_deviation"] = np.max( (v < self.v_lower) * (self.v_lower - v) )
        info["max_voltage_rise_deviation"] = np.max( (v > self.v_upper) * (v - self.v_upper) )
        # line loss
        line_loss = np.sum(self.powergrid.res_line["pl_mw"])
        avg_line_loss = np.mean(self.powergrid.res_line["pl_mw"])
        info["total_line_loss"] = line_loss
        # reactive power loss
        q = self.powergrid.res_sgen["q_mvar"].sort_index().to_numpy(copy=True)
        q_loss = np.mean(np.abs(q))
        info["q_loss"] = q_loss
        # reward function
        if self.reward_type == "l1":
            v_loss = np.mean(self.l1_loss(v)) * self.voltage_weight
        elif self.reward_type == "bowl":
            v_loss = np.mean(self.bowl_loss(v)) * self.voltage_weight
        elif self.reward_type == "l2":
            v_loss = np.mean(self.l2_loss(v)) * self.voltage_weight
        elif self.reward_type == "liu":
            v_loss = np.mean(self.liu_loss(v, self.v_lower, self.v_upper)) * self.voltage_weight
        elif self.reward_type == "bump":
            v_loss = np.mean(self.bump_loss(v)) * self.voltage_weight
        # print (f"This is the line_weight: {type(self.line_weight)}")
        # print (f"This is the q_weight: {self.q_weight}")
        if self.line_weight != None:
            loss = avg_line_loss * self.line_weight + v_loss
        elif self.q_weight != None:
            loss = q_loss * self.q_weight + v_loss
        else:
            raise NotImplementedError("Please at least give one weight, either q_weight or line_weight.")
        # dv/dq
        q = self.powergrid.res_sgen["q_mvar"].sort_index().to_numpy(copy=True)
        v_diff = np.repeat(np.expand_dims(v - self.last_v, axis=1), q.shape[0], axis=1) # (v_diff_dim, q_diff_dim)
        q_diff = np.repeat(np.expand_dims(q - self.last_q, axis=0), v.shape[0], axis=0) # (v_diff_dim, q_diff_dim)
        dv_dq = v_diff / (q_diff + 1e-7)
        norm_dv_dq = np.clip(np.mean(np.abs(dv_dq)), 0, 100)
        self.last_v = v
        self.last_q = q
        info["dv_dq"] = norm_dv_dq
        if "dv_dq" in self.reward_type:
            loss += -norm_dv_dq * self.dv_dq_weight
        info["destroy"] = 0.0
        return -loss, info
    
    def l1_loss(self, vs, v_ref=1.0):
        def _l1_loss(v):
            return np.abs( v - v_ref )
        return np.array([_l1_loss(v) for v in vs])

    def l2_loss(self, vs, v_ref=1.0):
        def _l2_loss(v):
            return 2 * np.square(v - v_ref)
        return np.array([_l2_loss(v) for v in vs])

    def liu_loss(self, vs, v_lower, v_upper):
        def _liu_loss(v):
            return np.square(max(0, v - v_upper)) + np.square(max(0, v_lower - v))
        return np.array([_liu_loss(v) for v in vs])

    def bowl_loss(self, vs, v_ref=1.0, scale=.1):
        def normal(v, loc, scale):
            return 1 / np.sqrt(2 * np.pi * scale**2) * np.exp( - 0.5 * np.square(v - loc) / scale**2 )
        def _bowl_loss(v):
            if np.abs(v-v_ref) > 0.05:
                return 2 * np.abs(v-v_ref) - 0.095
            else:
                return - 0.01 * normal(v, v_ref, scale) + 0.04
        return np.array([_bowl_loss(v) for v in vs])

    def bump_loss(self, vs):
        def _bump(v):
            if np.abs(v) < 1:
                return np.exp( - 1 / (1 - v**4) )
            elif 1 < v < 3:
                return np.exp( - 1 / (1 - ( v - 2 )**4 ) )
            else:
                return 0.0
        return np.array([_bump(v) for v in vs])

    def _get_res_bus_v(self):
        v = self.powergrid.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
        return v
    
    def _get_res_bus_active(self):
        active = self.powergrid.res_bus["p_mw"].sort_index().to_numpy(copy=True)
        return active

    def _get_res_bus_reactive(self):
        reactive = self.powergrid.res_bus["q_mvar"].sort_index().to_numpy(copy=True)
        return reactive

    def _get_res_line_loss(self):
        line_loss = self.powergrid.res_line["pl_mw"].sort_index().to_numpy(copy=True)
        return line_loss

    def _get_sgen_active(self):
        active = self.powergrid.sgen["p_mw"].to_numpy(copy=True)
        return active
    
    def _get_sgen_reactive(self):
        reactive = self.powergrid.sgen["q_mvar"].to_numpy(copy=True)
        return reactive