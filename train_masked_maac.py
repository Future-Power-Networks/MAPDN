import argparse
import os

import torch as th
import yaml
from tensorboardX import SummaryWriter

from environments.var_voltage_control.voltage_control_env import VoltageControl
from models.maac_masked import MaskedMAAC
from utilities.trainer import PGTrainer
from utilities.util import convert, dict2str


VALID_ATTENTION_MODES = ["full", "mask", "mask_prior"]
VALID_PRIOR_BIAS_MODES = ["add", "log"]


def build_parser():
    parser = argparse.ArgumentParser(description="Train MAPDN with structured MAAC critic.")
    parser.add_argument("--save-path", type=str, nargs="?", default="./")
    parser.add_argument("--env", type=str, nargs="?", default="var_voltage_control")
    parser.add_argument("--alias", type=str, nargs="?", default="")
    parser.add_argument("--mode", type=str, nargs="?", default="distributed")
    parser.add_argument("--scenario", type=str, nargs="?", default="case33_3min_final")
    parser.add_argument("--voltage-barrier-type", type=str, nargs="?", default="l1")
    parser.add_argument("--graph-dir", type=str, default=None)
    parser.add_argument("--graph-mask-path", type=str, default=None)
    parser.add_argument("--edge-prior-path", type=str, default=None)
    parser.add_argument("--attention-mode", type=str, choices=VALID_ATTENTION_MODES, default=None)
    parser.add_argument("--edge-prior-scale", type=float, default=None)
    parser.add_argument("--prior-bias-mode", type=str, choices=VALID_PRIOR_BIAS_MODES, default=None)
    parser.add_argument("--prior-bias-eps", type=float, default=None)
    parser.add_argument("--symmetrize-mask", action="store_true")
    parser.add_argument("--symmetrize-prior", action="store_true")
    parser.add_argument("--disable-full-attention-fallback", action="store_true")
    parser.add_argument("--alg-config", type=str, default="./args/alg_args/maac_masked.yaml")
    return parser


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_experiment_tag(cfg):
    mode = cfg["attention_mode"]
    if mode == "full":
        return "fullattn"
    if mode == "mask":
        return "maskonly"
    scale = str(cfg["edge_prior_scale"]).replace(".", "p")
    return f"maskprior-{cfg['prior_bias_mode']}-s{scale}"


def main():
    parser = build_parser()
    argv = parser.parse_args()

    env_config_dict = load_yaml(f"./args/env_args/{argv.env}.yaml")["env_args"]
    data_path = env_config_dict["data_path"].split("/")
    data_path[-1] = argv.scenario
    env_config_dict["data_path"] = "/".join(data_path)
    net_topology = argv.scenario

    assert net_topology in ["case33_3min_final", "case141_3min_final", "case322_3min_final"], (
        f"{net_topology} is not a valid scenario."
    )
    if argv.scenario == "case33_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8
    elif argv.scenario == "case141_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.6
    elif argv.scenario == "case322_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8

    assert argv.mode in ["distributed", "decentralised"]
    env_config_dict["mode"] = argv.mode
    env_config_dict["voltage_barrier_type"] = argv.voltage_barrier_type

    default_config_dict = load_yaml("./args/default.yaml")
    alg_config_dict = load_yaml(argv.alg_config)["alg_args"]
    alg_config_dict["action_scale"] = env_config_dict["action_scale"]
    alg_config_dict["action_bias"] = env_config_dict["action_bias"]

    if argv.graph_dir is not None:
        alg_config_dict["graph_dir"] = argv.graph_dir
    if argv.graph_mask_path is not None:
        alg_config_dict["graph_mask_path"] = argv.graph_mask_path
    if argv.edge_prior_path is not None:
        alg_config_dict["edge_prior_path"] = argv.edge_prior_path
    if argv.attention_mode is not None:
        alg_config_dict["attention_mode"] = argv.attention_mode
    if argv.edge_prior_scale is not None:
        alg_config_dict["edge_prior_scale"] = argv.edge_prior_scale
    if argv.prior_bias_mode is not None:
        alg_config_dict["prior_bias_mode"] = argv.prior_bias_mode
    if argv.prior_bias_eps is not None:
        alg_config_dict["prior_bias_eps"] = argv.prior_bias_eps
    alg_config_dict["symmetrize_mask"] = argv.symmetrize_mask
    alg_config_dict["symmetrize_prior"] = argv.symmetrize_prior
    alg_config_dict["full_attention_fallback"] = not argv.disable_full_attention_fallback

    log_name = "-".join(
        [
            argv.env,
            net_topology,
            argv.mode,
            "maac_masked",
            build_experiment_tag(alg_config_dict),
            argv.voltage_barrier_type,
            argv.alias,
        ]
    ).strip("-")
    alg_config_dict = {**default_config_dict, **alg_config_dict}

    env = VoltageControl(env_config_dict)
    alg_config_dict["agent_num"] = env.get_num_of_agents()
    alg_config_dict["obs_size"] = env.get_obs_size()
    alg_config_dict["action_dim"] = env.get_total_actions()
    args = convert(alg_config_dict)

    save_path = argv.save_path if argv.save_path.endswith("/") else argv.save_path + "/"
    ensure_dir(save_path)
    ensure_dir(save_path + "model_save")
    ensure_dir(save_path + "tensorboard")
    ensure_dir(save_path + "model_save/" + log_name)
    tb_dir = save_path + "tensorboard/" + log_name
    ensure_dir(tb_dir)
    for f in os.listdir(tb_dir):
        file_path = os.path.join(tb_dir, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    logger = SummaryWriter(tb_dir)
    print(f"{args}\n")
    train = PGTrainer(args, MaskedMAAC, env, logger)

    with open(tb_dir + "/log.txt", "w+", encoding="utf-8") as file:
        alg_args2str = dict2str(alg_config_dict, "alg_params")
        env_args2str = dict2str(env_config_dict, "env_params")
        file.write(alg_args2str + "\n")
        file.write(env_args2str + "\n")

    for i in range(args.train_episodes_num):
        stat = {}
        train.run(stat, i)
        train.logging(stat)
        if i % args.save_model_freq == args.save_model_freq - 1:
            train.print_info(stat)
            th.save(
                {"model_state_dict": train.behaviour_net.state_dict()},
                save_path + "model_save/" + log_name + "/model.pt",
            )
            print("The model is saved!\n")
    logger.close()


if __name__ == "__main__":
    main()
