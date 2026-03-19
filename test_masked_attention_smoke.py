import os
from types import SimpleNamespace

import numpy as np
import torch as th

from critics.masked_maac_critic import MaskedAttentionCritic


def build_args(attention_mode="mask_prior", edge_prior_scale=0.3, prior_bias_mode="add"):
    tmp_dir = "/tmp/masked_maac_smoke"
    os.makedirs(tmp_dir, exist_ok=True)

    mask = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )
    prior = np.array(
        [
            [0.0, 0.9, 0.0],
            [0.2, 0.0, 1.0],
            [0.0, 0.7, 0.0],
        ],
        dtype=np.float32,
    )
    mask_path = os.path.join(tmp_dir, "adj_mask.npy")
    prior_path = os.path.join(tmp_dir, "edge_prior.npy")
    np.save(mask_path, mask)
    np.save(prior_path, prior)

    return SimpleNamespace(
        hid_size=16,
        attend_heads=1,
        obs_size=5,
        action_dim=1,
        agent_num=3,
        continuous=True,
        norm_in=False,
        graph_mask_path=mask_path,
        edge_prior_path=prior_path,
        graph_dir=None,
        attention_mode=attention_mode,
        edge_prior_scale=edge_prior_scale,
        prior_bias_mode=prior_bias_mode,
        prior_bias_eps=1e-6,
        mask_fill_value=-1e9,
        full_attention_fallback=False,
        symmetrize_mask=False,
        symmetrize_prior=False,
    )


def test_forward_smoke():
    args = build_args()
    critic = MaskedAttentionCritic(args)
    batch_size = 4
    states = [th.randn(batch_size, args.obs_size) for _ in range(args.agent_num)]
    actions = [th.randn(batch_size, args.action_dim) for _ in range(args.agent_num)]
    sa = [th.cat((s, a), dim=-1) for s, a in zip(states, actions)]

    out = critic((states, actions, sa))
    assert len(out) == args.agent_num
    for agent_rets in out:
        q, reg = agent_rets
        assert q.shape == (batch_size, 1)
        assert reg.shape == (1, 1)


def test_modes_apply_expected_structure():
    base_logits = th.tensor([[[0.2, 0.2]]], dtype=th.float32)

    critic_full = MaskedAttentionCritic(build_args(attention_mode="full", edge_prior_scale=0.0))
    out_full = critic_full._apply_attention_structure(base_logits.clone(), agent_index=1)
    assert th.allclose(out_full, base_logits)

    critic_mask = MaskedAttentionCritic(build_args(attention_mode="mask", edge_prior_scale=0.0))
    out_mask = critic_mask._apply_attention_structure(base_logits.clone(), agent_index=0)
    assert out_mask[0, 0, 1] < -1e8
    assert th.isclose(out_mask[0, 0, 0], base_logits[0, 0, 0])

    critic_prior = MaskedAttentionCritic(build_args(attention_mode="mask_prior", edge_prior_scale=0.5))
    out_prior = critic_prior._apply_attention_structure(base_logits.clone(), agent_index=1)
    assert out_prior[0, 0, 1] > out_prior[0, 0, 0]

    critic_log = MaskedAttentionCritic(
        build_args(attention_mode="mask_prior", edge_prior_scale=0.5, prior_bias_mode="log")
    )
    out_log = critic_log._apply_attention_structure(base_logits.clone(), agent_index=1)
    assert out_log[0, 0, 1] > out_log[0, 0, 0]


if __name__ == "__main__":
    test_forward_smoke()
    test_modes_apply_expected_structure()
    print("masked attention smoke tests passed")
