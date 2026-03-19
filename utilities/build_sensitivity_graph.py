import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import pandapower as pp
import yaml

from environments.var_voltage_control.voltage_control_env import VoltageControl
from utilities.graph_utils import (
    DEFAULT_MASK_FILENAME,
    DEFAULT_PRIOR_FILENAME,
    DEFAULT_RANK_FILENAME,
    save_graph_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a sensitivity graph for MAPDN distributed PV agents."
    )
    parser.add_argument("--env", type=str, default="var_voltage_control")
    parser.add_argument("--scenario", type=str, default="case33_3min_final")
    parser.add_argument("--mode", type=str, default="distributed")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delta-q-ratio", type=float, default=0.05)
    parser.add_argument("--min-delta-q", type=float, default=1e-3)
    parser.add_argument("--metric", type=str, default="bus_voltage",
                        choices=["bus_voltage", "zone_mean_voltage", "zone_max_violation"])
    parser.add_argument("--threshold-quantile", type=float, default=0.75)
    parser.add_argument("--prior-stat", type=str, default="p75", choices=["mean", "p75"])
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--symmetrize", action="store_true")
    parser.add_argument("--voltage-barrier-type", type=str, default="l1")
    return parser.parse_args()


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(env_name: str, scenario: str, mode: str, voltage_barrier_type: str):
    env_config_dict = load_yaml(f"./args/env_args/{env_name}.yaml")["env_args"]
    data_path = env_config_dict["data_path"].split("/")
    data_path[-1] = scenario
    env_config_dict["data_path"] = "/".join(data_path)

    assert scenario in ["case33_3min_final", "case141_3min_final", "case322_3min_final"]
    if scenario == "case33_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8
    elif scenario == "case141_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.6
    elif scenario == "case322_3min_final":
        env_config_dict["action_bias"] = 0.0
        env_config_dict["action_scale"] = 0.8

    env_config_dict["mode"] = mode
    env_config_dict["voltage_barrier_type"] = voltage_barrier_type
    env = VoltageControl(env_config_dict)
    if env.args.mode != "distributed":
        raise ValueError("This graph builder is intended for MAPDN distributed mode only.")
    return env, env_config_dict


def build_agent_mappings(env: VoltageControl) -> Tuple[np.ndarray, List[str], List[List[int]]]:
    base = env.base_powergrid
    agent_buses = base.sgen["bus"].to_numpy(copy=True)
    agent_zones = list(base.sgen["name"].to_numpy(copy=True))
    zone_bus_ids = []
    for zone in agent_zones:
        buses = base.bus.index[base.bus["zone"] == zone].to_list()
        zone_bus_ids.append(buses)
    return agent_buses, agent_zones, zone_bus_ids


def index_to_day_hour_interval(env: VoltageControl, absolute_index: int) -> Tuple[int, int, int]:
    intervals_per_hour = 60 // env.time_delta
    intervals_per_day = 24 * intervals_per_hour
    day = absolute_index // intervals_per_day
    rem = absolute_index % intervals_per_day
    hour = rem // intervals_per_hour
    interval = rem % intervals_per_hour
    return int(day), int(hour), int(interval)


def reset_env_at_index(env: VoltageControl, absolute_index: int) -> None:
    day, hour, interval = index_to_day_hour_interval(env, absolute_index)
    env.manual_reset(day=day, hour=hour, interval=interval)
    env.powergrid.sgen["q_mvar"] = 0.0
    pp.runpp(env.powergrid)


def compute_metric_vector(
    net,
    metric: str,
    agent_buses: np.ndarray,
    zone_bus_ids: List[List[int]],
    v_lower: float,
    v_upper: float,
) -> np.ndarray:
    vm = net.res_bus["vm_pu"].sort_index()
    values = []
    for agent_idx, bus_id in enumerate(agent_buses):
        zone_buses = zone_bus_ids[agent_idx]
        zone_vm = vm.loc[zone_buses].to_numpy(copy=True)
        if metric == "bus_voltage":
            values.append(float(vm.loc[bus_id]))
        elif metric == "zone_mean_voltage":
            values.append(float(zone_vm.mean()))
        elif metric == "zone_max_violation":
            low = np.maximum(v_lower - zone_vm, 0.0)
            high = np.maximum(zone_vm - v_upper, 0.0)
            values.append(float(max(low.max(initial=0.0), high.max(initial=0.0))))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return np.asarray(values, dtype=np.float32)


def perturb_and_solve(env: VoltageControl, base_net, agent_j: int, delta_q: float):
    trial = copy.deepcopy(base_net)
    q_cap = float(np.sqrt(env.s_max[agent_j] ** 2 - trial.sgen["p_mw"].iloc[agent_j] ** 2))
    q0 = float(trial.sgen["q_mvar"].iloc[agent_j])
    q1 = float(np.clip(q0 + delta_q, -q_cap, q_cap))
    effective_delta = q1 - q0
    trial.sgen.iat[agent_j, trial.sgen.columns.get_loc("q_mvar")] = q1
    pp.runpp(trial)
    return trial, effective_delta


def compute_sample_sensitivity(
    env: VoltageControl,
    metric: str,
    agent_buses: np.ndarray,
    zone_bus_ids: List[List[int]],
    delta_q_ratio: float,
    min_delta_q: float,
) -> np.ndarray:
    base_net = copy.deepcopy(env.powergrid)
    base_metric = compute_metric_vector(
        net=base_net,
        metric=metric,
        agent_buses=agent_buses,
        zone_bus_ids=zone_bus_ids,
        v_lower=env.v_lower,
        v_upper=env.v_upper,
    )

    n_agents = env.get_num_of_agents()
    sensitivity = np.zeros((n_agents, n_agents), dtype=np.float32)
    for agent_j in range(n_agents):
        q_cap = float(np.sqrt(env.s_max[agent_j] ** 2 - base_net.sgen["p_mw"].iloc[agent_j] ** 2))
        delta_q = max(min_delta_q, delta_q_ratio * q_cap)

        plus_net, plus_dq = perturb_and_solve(env, base_net, agent_j, +delta_q)
        minus_net, minus_dq = perturb_and_solve(env, base_net, agent_j, -delta_q)

        plus_metric = compute_metric_vector(
            plus_net, metric, agent_buses, zone_bus_ids, env.v_lower, env.v_upper
        )
        minus_metric = compute_metric_vector(
            minus_net, metric, agent_buses, zone_bus_ids, env.v_lower, env.v_upper
        )
        denom = plus_dq - minus_dq
        if abs(denom) < 1e-12:
            continue
        sensitivity[:, agent_j] = (plus_metric - minus_metric) / denom

    # self edges are intentionally suppressed because MAAC excludes self-attention.
    np.fill_diagonal(sensitivity, 0.0)
    return sensitivity


def build_mask_from_scores(scores: np.ndarray, top_k: int, threshold_quantile: float) -> np.ndarray:
    n_agents = scores.shape[0]
    mask = np.zeros_like(scores, dtype=np.float32)
    for i in range(n_agents):
        row = scores[i].copy()
        row[i] = 0.0
        positive = row[row > 0]
        if positive.size > 0:
            threshold = float(np.quantile(positive, threshold_quantile))
            selected = row >= threshold
        else:
            selected = np.zeros(n_agents, dtype=bool)
        selected[i] = False

        if top_k > 0:
            order = np.argsort(row)[::-1]
            count = 0
            for j in order:
                if j == i:
                    continue
                selected[j] = True
                count += 1
                if count >= top_k:
                    break
        mask[i, selected] = 1.0
        mask[i, i] = 0.0
    return mask


def build_rank_matrix(scores: np.ndarray) -> np.ndarray:
    n_agents = scores.shape[0]
    ranks = np.full((n_agents, n_agents - 1), -1, dtype=np.int64)
    for i in range(n_agents):
        row = scores[i].copy()
        row[i] = -np.inf
        order = np.argsort(row)[::-1]
        order = order[order != i]
        ranks[i, : len(order)] = order
    return ranks


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    env, env_config = build_env(
        env_name=args.env,
        scenario=args.scenario,
        mode=args.mode,
        voltage_barrier_type=args.voltage_barrier_type,
    )
    agent_buses, agent_zones, zone_bus_ids = build_agent_mappings(env)
    n_agents = env.get_num_of_agents()

    total_points = len(env.pv_data)
    reserve = env.episode_limit + env.history + 2
    max_start = max(1, total_points - reserve)
    sample_size = min(args.num_samples, max_start)
    sample_indices = rng.choice(max_start, size=sample_size, replace=False)

    sensitivities = []
    failures = 0
    for idx in sample_indices:
        try:
            reset_env_at_index(env, int(idx))
            sample_sens = compute_sample_sensitivity(
                env=env,
                metric=args.metric,
                agent_buses=agent_buses,
                zone_bus_ids=zone_bus_ids,
                delta_q_ratio=args.delta_q_ratio,
                min_delta_q=args.min_delta_q,
            )
            sensitivities.append(sample_sens)
        except Exception as exc:  # pragma: no cover - best-effort data scan
            failures += 1
            print(f"[WARN] skip index {idx}: {exc}")

    if not sensitivities:
        raise RuntimeError("No valid sensitivity samples were collected.")

    sens_stack = np.stack(sensitivities, axis=0)
    abs_stack = np.abs(sens_stack)
    mean_abs = abs_stack.mean(axis=0)
    p75_abs = np.quantile(abs_stack, 0.75, axis=0)
    var_abs = abs_stack.var(axis=0)

    score = p75_abs if args.prior_stat == "p75" else mean_abs
    mask = build_mask_from_scores(score, top_k=args.top_k, threshold_quantile=args.threshold_quantile)
    prior = score.copy()
    np.fill_diagonal(prior, 0.0)
    prior = prior * mask
    row_max = np.maximum(prior.max(axis=1, keepdims=True), 1e-12)
    prior = prior / row_max
    ranks = build_rank_matrix(score)

    if args.symmetrize:
        mask = np.maximum(mask, mask.T)
        prior = np.maximum(prior, prior.T)
        np.fill_diagonal(mask, 0.0)
        np.fill_diagonal(prior, 0.0)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, DEFAULT_MASK_FILENAME), mask.astype(np.float32))
    np.save(os.path.join(args.output_dir, DEFAULT_PRIOR_FILENAME), prior.astype(np.float32))
    np.save(os.path.join(args.output_dir, DEFAULT_RANK_FILENAME), ranks.astype(np.int64))
    np.savez(
        os.path.join(args.output_dir, "sensitivity_stats.npz"),
        mean_abs=mean_abs.astype(np.float32),
        p75_abs=p75_abs.astype(np.float32),
        var_abs=var_abs.astype(np.float32),
        raw_abs=abs_stack.astype(np.float32),
    )
    density = float(mask.sum() / max(n_agents * (n_agents - 1), 1))

    save_graph_metadata(
        args.output_dir,
        {
            "scenario": args.scenario,
            "mode": args.mode,
            "metric": args.metric,
            "num_agents": int(n_agents),
            "agent_zones": agent_zones,
            "num_requested_samples": int(args.num_samples),
            "num_valid_samples": int(len(sensitivities)),
            "num_failed_samples": int(failures),
            "delta_q_ratio": float(args.delta_q_ratio),
            "min_delta_q": float(args.min_delta_q),
            "threshold_quantile": float(args.threshold_quantile),
            "top_k": int(args.top_k),
            "prior_stat": args.prior_stat,
            "symmetrize": bool(args.symmetrize),
            "graph_density": density,
            "v_lower": float(env.v_lower),
            "v_upper": float(env.v_upper),
            "env_config": env_config,
        },
    )

    print(f"Saved graph to: {args.output_dir}")
    print(f"adj_mask shape: {mask.shape}")
    print(f"edge_prior shape: {prior.shape}")
    print(f"collected samples: {len(sensitivities)} / {args.num_samples}")
    print(f"graph density: {density:.4f}")


if __name__ == "__main__":
    main()
