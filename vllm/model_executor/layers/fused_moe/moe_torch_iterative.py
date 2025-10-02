# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.fused_moe import (
    activation_without_mul, fused_topk)


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    override_config=None,
    use_grouped_topk: bool = False,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor = None,
) -> torch.Tensor:
    """
    This function is a pure python version of the fused_moe kernel.
    It is used for debugging and testing only.
    """
    #
    assert not inplace, "inplace=True is not supported in the pure python version"
    num_tokens, hidden_dim = hidden_states.shape
    num_experts = w1.shape[0]
    intermediate_size = w1.shape[2]
    # Get top-k experts for each token
    topk_weights, topk_ids, _ = fused_topk(hidden_states, gating_output, topk,
                                           renormalize, use_grouped_topk,
                                           num_expert_group, topk_group,
                                           scoring_func,
                                           routed_scaling_factor,
                                           e_score_correction_bias)
    final_hidden_states = torch.zeros_like(hidden_states)
    for i in range(num_tokens):
        for j in range(topk):
            expert_id = topk_ids[i, j].item()
            w1_expert = w1[expert_id, :, :]
            w2_expert = w2[expert_id, :, :]
            # gate_proj and up_proj
            gate_up = F.linear(hidden_states[i], w1_expert)
            # activation function
            gate, up = gate_up.chunk(2, dim=-1)
            gate = F.silu(gate)
            # element wise multiplication
            intermediate = gate * up
            # down_proj
            down = F.linear(intermediate, w2_expert)
            # multiply by routing weight
            down = down * topk_weights[i, j]
            # add to final hidden states
            final_hidden_states[i] += down
    return final_hidden_states
