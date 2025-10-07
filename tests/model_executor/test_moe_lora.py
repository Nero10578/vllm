import os
import pytest
import torch

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.moe_torch_iterative import fused_moe as fused_moe_torch_iterative

@pytest.mark.parametrize("num_tokens", [1, 2, 128])
@pytest.mark.parametrize("hidden_dim", [256, 512])
@pytest.mark.parametrize("num_experts", [4, 8])
@pytest.mark.parametrize("top_k", [2])
def test_moe_lora_equivalence(num_tokens, hidden_dim, num_experts, top_k):
    """
    Tests that the iterative MoE implementation is numerically equivalent to the
    original fused MoE kernel.
    """
    torch.manual_seed(0)
    hidden_states = torch.randn(num_tokens, hidden_dim).cuda()
    w1 = torch.randn(num_experts, 2 * hidden_dim, hidden_dim).cuda()
    w2 = torch.randn(num_experts, hidden_dim, hidden_dim).cuda()
    gating_output = torch.randn(num_tokens, num_experts).cuda()

    # Run with original fused kernel
    os.environ["VLLM_ENABLE_LORA_ON_MOE"] = "0"
    fused_output = fused_experts(
        hidden_states=hidden_states.clone(),
        w1=w1.clone(),
        w2=w2.clone(),
        topk_weights=gating_output.clone(),
        topk_ids=torch.randint(0, num_experts, (num_tokens, top_k)).cuda(),
        inplace=False,
    )

    # Run with iterative kernel
    os.environ["VLLM_ENABLE_LORA_ON_MOE"] = "1"
    iterative_output = fused_moe_torch_iterative(
        hidden_states=hidden_states.clone(),
        w1=w1.clone(),
        w2=w2.clone(),
        gating_output=gating_output.clone(),
        topk=top_k,
        renormalize=True,
    )

    assert torch.allclose(fused_output, iterative_output, atol=1e-5, rtol=1e-5)
