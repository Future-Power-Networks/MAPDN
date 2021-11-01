import torch as th
import os
import argparse
import yaml
from tensorboardX import SummaryWriter

from models.model_registry import Model, Strategy
from environments.var_voltage_control.voltage_control_env import VoltageControl
from utilities.util import convert, dict2str
from utilities.trainer import PGTrainer



parser = argparse.ArgumentParser(description="Train rl agent.")
parser.add_argument("--save-path", type=str, nargs="?", default="./", help="Please enter the directory of saving model.")
parser.add_argument("--alg", type=str, nargs="?", default="maddpg", help="Please enter the alg name.")
parser.add_argument("--env", type=str, nargs="?", default="var_voltage_control", help="Please enter the env name.")
parser.add_argument("--alias", type=str, nargs="?", default="", help="Please enter the alias for exp control.")
parser.add_argument("--mode", type=str, nargs="?", default="distributed", help="Please enter the mode: distributed or decentralised.")
parser.add_argument("--scenario", type=str, nargs="?", default="bus33_3min_final", help="Please input the valid name of an environment scenario.")
parser.add_argument("--voltage-barrier-type", type=str, nargs="?", default="l1", help="Please input the valid voltage barrier type: l1, courant_beltrami, l2, bowl or bump.")
argv = parser.parse_args()

# load env args
with open("./args/env_args/"+argv.env+".yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]
data_path = env_config_dict["data_path"].split("/")
data_path[-1] = argv.scenario
env_config_dict["data_path"] = "/".join(data_path)
net_topology = argv.scenario

# set the action range
assert net_topology in ['case33_3min_final', 'case141_3min_final', 'case322_3min_final'], f'{net_topology} is not a valid scenario.'
if argv.scenario == 'case33_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.8
elif argv.scenario == 'case141_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.6
elif argv.scenario == 'case322_3min_final':
    env_config_dict["action_bias"] = 0.0
    env_config_dict["action_scale"] = 0.8

assert argv.mode in ['distributed', 'decentralised'], "Please input the correct mode, e.g. distributed or decentralised."
env_config_dict["mode"] = argv.mode
env_config_dict["voltage_barrier_type"] = argv.voltage_barrier_type

# load default args
with open("./args/default.yaml", "r") as f:
    default_config_dict = yaml.safe_load(f)

# load alg args
with open("./args/alg_args/" + argv.alg + ".yaml", "r") as f:
    alg_config_dict = yaml.safe_load(f)["alg_args"]
    alg_config_dict["action_scale"] = env_config_dict["action_scale"]
    alg_config_dict["action_bias"] = env_config_dict["action_bias"]

log_name = "-".join([argv.env, net_topology, argv.mode, argv.alg, argv.voltage_barrier_type, argv.alias])
alg_config_dict = {**default_config_dict, **alg_config_dict}

# define envs
env = VoltageControl(env_config_dict)

alg_config_dict["agent_num"] = env.get_num_of_agents()
alg_config_dict["obs_size"] = env.get_obs_size()
alg_config_dict["action_dim"] = env.get_total_actions()
args = convert(alg_config_dict)

# define the save path
if argv.save_path[-1] is "/":
    save_path = argv.save_path
else:
    save_path = argv.save_path+"/"

# create the save folders
if "model_save" not in os.listdir(save_path):
    os.mkdir(save_path + "model_save")
if "tensorboard" not in os.listdir(save_path):
    os.mkdir(save_path + "tensorboard")
if log_name not in os.listdir(save_path + "model_save/"):
    os.mkdir(save_path + "model_save/" + log_name)
if log_name not in os.listdir(save_path + "tensorboard/"):
    os.mkdir(save_path + "tensorboard/" + log_name)
else:
    path = save_path + "tensorboard/" + log_name
    for f in os.listdir(path):
        file_path = os.path.join(path,f)
        if os.path.isfile(file_path):
            os.remove(file_path)

# create the logger
logger = SummaryWriter(save_path + "tensorboard/" + log_name)

model = Model[argv.alg]

strategy = Strategy[argv.alg]

print (f"{args}\n")

if strategy == "pg":
    train = PGTrainer(args, model, env, logger)
elif strategy == "q":
    raise NotImplementedError("This needs to be implemented.")
else:
    raise RuntimeError("Please input the correct strategy, e.g. pg or q.")

with open(save_path + "tensorboard/" + log_name + "/log.txt", "w+") as file:
    alg_args2str = dict2str(alg_config_dict, 'alg_params')
    env_args2str = dict2str(env_config_dict, 'env_params')
    file.write(alg_args2str + "\n")
    file.write(env_args2str + "\n")

for i in range(args.train_episodes_num):
    stat = {}
    train.run(stat, i)
    train.logging(stat)
    if i%args.save_model_freq == args.save_model_freq-1:
        train.print_info(stat)
        th.save({"model_state_dict": train.behaviour_net.state_dict()}, save_path + "model_save/" + log_name + "/model.pt")
        print ("The model is saved!\n")

logger.close()
