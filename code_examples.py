from environments.var_voltage_control.voltage_control_env import VoltageControl
import numpy as np
import yaml


def main():
    # load env args
    with open("./args/env_args/var_voltage_control.yaml", "r") as f:
        env_config_dict = yaml.safe_load(f)["env_args"]
    data_path = env_config_dict["data_path"].split("/")
    net_topology = "case33_3min_final" # case33_3min_final / case141_3min_final / case322_3min_final
    data_path[-1] = net_topology 
    env_config_dict["data_path"] = "/".join(data_path)

    # set the action range
    assert net_topology in ['case33_3min_final', 'case141_3min_final', 'case322_3min_final'], f'{net_topology} is not a valid scenario.'
    if net_topology == 'case33_3min_final':
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8
    elif net_topology == 'case141_3min_final':
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.6
    elif net_topology == 'case322_3min_final':
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8
    
    # define control mode and voltage barrier function
    env_config_dict["mode"] = 'distributed'
    env_config_dict["voltage_barrier_type"] = 'l1'

    # define envs
    env = VoltageControl(env_config_dict)

    n_agents = env.get_num_of_agents()
    n_actions = env.get_total_actions()

    n_episodes = 10

    for e in range(n_episodes):
        state, global_state = env.reset()
        max_steps = 100
        episode_reward = 0

        for t in range(max_steps):
            obs = env.get_obs()
            state = env.get_state()

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.normal(0, 0.5, n_actions)
                action = action[avail_actions_ind]
                actions.append(action)

            actions = np.concatenate(actions, axis=0)
            
            reward, _, info = env.step(actions)

            episode_reward += reward

        print (f"Total reward in epsiode {e} = {episode_reward:.2f}")

    env.close()

if __name__ == '__main__':
    main()