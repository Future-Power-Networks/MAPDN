import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from utilities.graph_utils import load_graph


class MaskedAttentionCritic(nn.Module):
    """
    MAAC critic with configurable structural attention.

    Supported modes:
    - full:       reproduce vanilla MAAC full-attention.
    - mask:       sparse attention with a fixed binary mask.
    - mask_prior: sparse attention with a fixed binary mask plus a static prior bias.

    The prior is injected at the logit level so the final edge weights are still
    learned dynamically by the attention module.
    """

    VALID_ATTENTION_MODES = {"full", "mask", "mask_prior"}
    VALID_PRIOR_BIAS_MODES = {"add", "log"}

    def __init__(self, args):
        super(MaskedAttentionCritic, self).__init__()
        self.hidden_dim = args.hid_size
        self.attend_heads = args.attend_heads
        assert (self.hidden_dim % self.attend_heads) == 0

        self.sa_sizes = [(args.obs_size, args.action_dim)] * args.agent_num
        self.nagents = args.agent_num
        self.continuous = args.continuous

        self.attention_mode = str(getattr(args, "attention_mode", "mask_prior")).lower()
        if self.attention_mode not in self.VALID_ATTENTION_MODES:
            raise ValueError(
                f"Unsupported attention_mode={self.attention_mode}. "
                f"Choose from {sorted(self.VALID_ATTENTION_MODES)}."
            )

        self.edge_prior_scale = float(getattr(args, "edge_prior_scale", 0.0))
        self.prior_bias_mode = str(getattr(args, "prior_bias_mode", "add")).lower()
        if self.prior_bias_mode not in self.VALID_PRIOR_BIAS_MODES:
            raise ValueError(
                f"Unsupported prior_bias_mode={self.prior_bias_mode}. "
                f"Choose from {sorted(self.VALID_PRIOR_BIAS_MODES)}."
            )
        self.prior_bias_eps = float(getattr(args, "prior_bias_eps", 1e-6))
        self.mask_fill_value = float(getattr(args, "mask_fill_value", -1e9))

        mask, prior = load_graph(
            n_agents=args.agent_num,
            graph_mask_path=getattr(args, "graph_mask_path", None),
            edge_prior_path=getattr(args, "edge_prior_path", None),
            graph_dir=getattr(args, "graph_dir", None),
            full_attention_fallback=bool(getattr(args, "full_attention_fallback", True)),
            symmetrize_mask=bool(getattr(args, "symmetrize_mask", False)),
            symmetrize_prior=bool(getattr(args, "symmetrize_prior", False)),
        )
        self.register_buffer("full_adj_mask", th.tensor(mask, dtype=th.float32))
        self.register_buffer("full_edge_prior", th.tensor(prior, dtype=th.float32))

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.biases = nn.ModuleList()
        self.state_encoders = nn.ModuleList()

        for sdim, adim in self.sa_sizes:
            idim = sdim + adim
            odim = 1 if args.continuous else adim

            encoder = nn.Sequential()
            if args.norm_in:
                encoder.add_module("enc_bn", nn.BatchNorm1d(idim, affine=False))
            encoder.add_module("enc_fc1", nn.Linear(idim, self.hidden_dim))
            encoder.add_module("enc_nl", nn.LeakyReLU())
            self.critic_encoders.append(encoder)

            critic = nn.Sequential()
            critic.add_module("critic_fc1", nn.Linear(2 * self.hidden_dim, self.hidden_dim))
            critic.add_module("critic_nl", nn.LeakyReLU())
            critic.add_module("critic_fc2", nn.Linear(self.hidden_dim, odim))
            self.critics.append(critic)

            bias = nn.Sequential()
            bias.add_module("bias_fc1", nn.Linear(self.hidden_dim, self.hidden_dim))
            bias.add_module("bias_nl", nn.LeakyReLU())
            bias.add_module("bias_fc2", nn.Linear(self.hidden_dim, 1))
            self.biases.append(bias)

            state_encoder = nn.Sequential()
            if args.norm_in:
                state_encoder.add_module("s_enc_bn", nn.BatchNorm1d(sdim, affine=False))
            state_encoder.add_module("s_enc_fc1", nn.Linear(sdim, self.hidden_dim))
            state_encoder.add_module("s_enc_nl", nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = self.hidden_dim // self.attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for _ in range(self.attend_heads):
            self.key_extractors.append(nn.Linear(self.hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(self.hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(
                nn.Sequential(nn.Linear(self.hidden_dim, attend_dim), nn.LeakyReLU())
            )

    def _others_index(self, agent_index: int):
        return [j for j in range(self.nagents) if j != agent_index]

    def _get_mask_and_prior_rows(self, agent_index: int, batch_size: int, device):
        others = self._others_index(agent_index)
        mask_row = self.full_adj_mask[agent_index, others].view(1, 1, -1).to(device)
        prior_row = self.full_edge_prior[agent_index, others].view(1, 1, -1).to(device)
        return mask_row.expand(batch_size, 1, -1), prior_row.expand(batch_size, 1, -1)

    def _apply_attention_structure(self, scaled_attend_logits, agent_index: int):
        batch_size = scaled_attend_logits.shape[0]
        device = scaled_attend_logits.device
        mask_row, prior_row = self._get_mask_and_prior_rows(agent_index, batch_size, device)

        if self.attention_mode == "full":
            return scaled_attend_logits

        if (mask_row.sum(dim=-1) <= 0).any():
            mask_row = th.ones_like(mask_row)

        structured_logits = scaled_attend_logits.masked_fill(mask_row <= 0, self.mask_fill_value)

        if self.attention_mode == "mask":
            return structured_logits

        if self.edge_prior_scale == 0.0:
            return structured_logits

        if self.prior_bias_mode == "add":
            prior_bias = prior_row
        else:
            prior_bias = th.log(prior_row.clamp_min(self.prior_bias_eps))

        return structured_logits + self.edge_prior_scale * prior_bias

    def _masked_attention(self, selector, keys, values, agent_index):
        batch_size = selector.shape[0]
        if len(keys) == 0:
            head_dim = selector.shape[-1]
            zeros = selector.new_zeros((batch_size, head_dim))
            empty_logits = selector.new_zeros((batch_size, 1, 0))
            empty_probs = selector.new_zeros((batch_size, 1, 0))
            return zeros, empty_logits, empty_probs

        attend_logits = th.matmul(
            selector.view(batch_size, 1, -1),
            th.stack(keys).permute(1, 2, 0),
        )
        scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
        structured_logits = self._apply_attention_structure(scaled_attend_logits, agent_index)

        attend_weights = F.softmax(structured_logits, dim=2)
        other_values = (th.stack(values).permute(1, 2, 0) * attend_weights).sum(dim=2)
        return other_values, attend_logits, attend_weights

    def forward(self, inps, return_q=True, regularize=True):
        agents = range(len(self.critic_encoders))
        states, actions, sa = inps

        sa_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, sa)]
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]

        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        all_head_selectors = [
            [sel_ext(enc) for i, enc in enumerate(s_encodings) if i in agents]
            for sel_ext in self.selector_extractors
        ]

        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]

        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
            all_head_keys, all_head_values, all_head_selectors
        ):
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != a_i]
                values = [v for j, v in enumerate(curr_head_values) if j != a_i]
                other_values, attend_logits, attend_weights = self._masked_attention(
                    selector=selector,
                    keys=keys,
                    values=values,
                    agent_index=a_i,
                )
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)

        all_rets = []
        for i, a_i in enumerate(agents):
            agent_rets = []
            if self.continuous:
                critic_in = th.cat((sa_encodings[i], *other_all_values[i]), dim=1)
                all_q = self.critics[a_i](critic_in)
                q = all_q
            else:
                critic_in = th.cat((s_encodings[i], *other_all_values[i]), dim=1)
                all_q = self.critics[a_i](critic_in)
                int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
                q = all_q.gather(1, int_acs)

            bias_in = s_encodings[i]
            b = self.biases[a_i](bias_in)

            if return_q:
                agent_rets.append(q - b)
            if regularize:
                if len(all_attend_logits[i]) == 0:
                    attend_mag_reg = q.new_tensor(0.0)
                else:
                    attend_mag_reg = 1e-3 * sum((logit ** 2).mean() for logit in all_attend_logits[i])
                regs = attend_mag_reg.view(1, 1)
                agent_rets.append(regs)
            all_rets.append(agent_rets)
        return all_rets
