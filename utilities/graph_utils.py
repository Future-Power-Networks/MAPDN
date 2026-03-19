import json
import os
from typing import Optional, Tuple

import numpy as np


DEFAULT_MASK_FILENAME = "adj_mask.npy"
DEFAULT_PRIOR_FILENAME = "edge_prior.npy"
DEFAULT_RANK_FILENAME = "edge_rank.npy"


def fully_connected_mask(n_agents: int) -> np.ndarray:
    mask = np.ones((n_agents, n_agents), dtype=np.float32)
    np.fill_diagonal(mask, 0.0)
    return mask


def zero_prior(n_agents: int) -> np.ndarray:
    return np.zeros((n_agents, n_agents), dtype=np.float32)


def _ensure_square(name: str, matrix: np.ndarray, n_agents: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (n_agents, n_agents):
        raise ValueError(
            f"{name} must have shape {(n_agents, n_agents)}, got {matrix.shape}."
        )
    return matrix


def _safe_load_npy(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file does not exist: {path}")
    return np.load(path)


def normalize_prior(prior: np.ndarray, mask: np.ndarray) -> np.ndarray:
    prior = np.asarray(prior, dtype=np.float32)
    prior = np.abs(prior) * mask
    np.fill_diagonal(prior, 0.0)
    row_max = np.maximum(prior.max(axis=1, keepdims=True), 1e-12)
    return prior / row_max


def maybe_symmetrize(matrix: np.ndarray, enabled: bool, mode: str = "max") -> np.ndarray:
    if not enabled:
        return matrix
    if mode == "max":
        return np.maximum(matrix, matrix.T)
    if mode == "mean":
        return 0.5 * (matrix + matrix.T)
    raise ValueError(f"Unsupported symmetrize mode: {mode}")


def resolve_graph_paths(
    graph_dir: Optional[str],
    graph_mask_path: Optional[str],
    edge_prior_path: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if graph_dir:
        if graph_mask_path is None:
            graph_mask_path = os.path.join(graph_dir, DEFAULT_MASK_FILENAME)
        if edge_prior_path is None:
            edge_prior_path = os.path.join(graph_dir, DEFAULT_PRIOR_FILENAME)
    return graph_mask_path, edge_prior_path


def load_graph(
    n_agents: int,
    graph_mask_path: Optional[str] = None,
    edge_prior_path: Optional[str] = None,
    graph_dir: Optional[str] = None,
    full_attention_fallback: bool = True,
    symmetrize_mask: bool = False,
    symmetrize_prior: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    graph_mask_path, edge_prior_path = resolve_graph_paths(
        graph_dir=graph_dir,
        graph_mask_path=graph_mask_path,
        edge_prior_path=edge_prior_path,
    )

    mask = _safe_load_npy(graph_mask_path)
    if mask is None:
        if not full_attention_fallback:
            raise FileNotFoundError(
                "No adj_mask.npy was provided, and full_attention_fallback is False."
            )
        mask = fully_connected_mask(n_agents)
    mask = _ensure_square("adj_mask", mask, n_agents)
    mask = (mask > 0).astype(np.float32)
    np.fill_diagonal(mask, 0.0)
    mask = maybe_symmetrize(mask, symmetrize_mask, mode="max")

    prior = _safe_load_npy(edge_prior_path)
    if prior is None:
        prior = zero_prior(n_agents)
    prior = _ensure_square("edge_prior", prior, n_agents)
    prior = maybe_symmetrize(prior, symmetrize_prior, mode="mean")
    prior = normalize_prior(prior, mask)
    return mask.astype(np.float32), prior.astype(np.float32)


def graph_density(mask: np.ndarray) -> float:
    n_agents = mask.shape[0]
    max_edges = max(n_agents * (n_agents - 1), 1)
    return float(mask.sum() / max_edges)


def save_graph_metadata(output_dir: str, metadata: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "graph_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
