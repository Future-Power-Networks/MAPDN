import torch as th
import os
import argparse
import yaml
from tensorboardX import SummaryWriter

from models.model_registry import Model, Strategy
from environments.var_voltage_control.voltage_control_env import VoltageControl
from utilities.util import convert, dict2str, setup_voltage_control_scenario
from utilities.trainer import PGTrainer



parser = argparse.ArgumentParser(description="Train rl agent.")
parser.add_argument("--save-path", type=str, nargs="?", default="./", help="Please enter the directory of saving model.")
parser.add_argument("--alg", type=str, nargs="?", default="maddpg", help="Please enter the alg name.")
parser.add_argument("--env", type=str, nargs="?", default="var_voltage_control", help="Please enter the env name.")
parser.add_argument("--alias", type=str, nargs="?", default="", help="Please enter the alias for exp control.")
parser.add_argument("--mode", type=str, nargs="?", default="distributed", help="Please enter the mode: distributed or decentralised.")
parser.add_argument("--scenario", type=str, nargs="?", default="case33_3min_final", help="Please input the valid name of an environment scenario.")
parser.add_argument("--voltage-barrier-type", type=str, nargs="?", default="l1", help="Please input the valid voltage barrier type: l1, courant_beltrami, l2, bowl or bump.")
argv = parser.parse_args()

# load env args
with open("./args/env_args/"+argv.env+".yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]
net_topology = argv.scenario

setup_voltage_control_scenario(env_config_dict, net_topology)

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
save_path = os.path.normpath(argv.save_path)
model_save_root = os.path.join(save_path, "model_save")
tensorboard_root = os.path.join(save_path, "tensorboard")
model_save_dir = os.path.join(model_save_root, log_name)
tensorboard_dir = os.path.join(tensorboard_root, log_name)

# create the save folders
if "model_save" not in os.listdir(save_path):
    os.mkdir(model_save_root)
if "tensorboard" not in os.listdir(save_path):
    os.mkdir(tensorboard_root)
if log_name not in os.listdir(model_save_root):
    os.mkdir(model_save_dir)
if log_name not in os.listdir(tensorboard_root):
    os.mkdir(tensorboard_dir)
else:
    path = tensorboard_dir
    for f in os.listdir(path):
        file_path = os.path.join(path,f)
        if os.path.isfile(file_path):
            os.remove(file_path)

# create the logger
logger = SummaryWriter(tensorboard_dir)

model = Model[argv.alg]

strategy = Strategy[argv.alg]

print (f"{args}\n")

if strategy == "pg":
    train = PGTrainer(args, model, env, logger)
elif strategy == "q":
    raise NotImplementedError("This needs to be implemented.")
else:
    raise RuntimeError("Please input the correct strategy, e.g. pg or q.")

with open(os.path.join(tensorboard_dir, "log.txt"), "w+") as file:
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
        th.save({"model_state_dict": train.behaviour_net.state_dict()}, os.path.join(model_save_dir, "model.pt"))
        print ("The model is saved!\n")

logger.close()
