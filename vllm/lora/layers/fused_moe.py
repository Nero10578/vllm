# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, List, Callable, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config.lora import LoRAConfig

from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.moe_torch_iterative import fused_moe


class FusedMoEWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: FusedMoE) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.num_experts = base_layer.local_num_experts
        self.lora_a_stacked = [None] * 1 # Assuming 1 slot for now
        self.lora_b_stacked = [None] * 1

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: "LoRAConfig",
        model_config: Optional["PretrainedConfig"] = None,
    ) -> None:
        # We don't create weights here because they are packed and managed by LoRAModelManager
        # But we need to initialize the storage
        self.lora_a_stacked = [None] * max_loras
        self.lora_b_stacked = [None] * max_loras

    def reset_lora(self, index: int):
        self.lora_a_stacked[index] = None
        self.lora_b_stacked[index] = None

    def set_lora(
        self,
        index: int,
        lora_a: List[torch.Tensor],
        lora_b: List[torch.Tensor],
        embeddings_tensor: Optional[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
    ):
        # lora_a and lora_b are lists of tensors corresponding to the packed modules
        # Order: [experts.0.gate_up_proj, experts.0.down_proj, experts.1.gate_up_proj, ...]
        self.lora_a_stacked[index] = lora_a
        self.lora_b_stacked[index] = lora_b

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: "LoRAConfig",
        packed_modules_list: list,
        model_config: Optional["PretrainedConfig"],
    ) -> bool:
        return isinstance(source_layer, FusedMoE)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        # This is the forward method called by the model
        # It delegates to apply() which handles the LoRA logic
        return self.apply(
            layer=self.base_layer,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.base_layer.top_k,
            renormalize=self.base_layer.renormalize,
            use_grouped_topk=self.base_layer.use_grouped_topk,
            topk_group=self.base_layer.topk_group,
            num_expert_group=self.base_layer.num_expert_group,
            global_num_experts=self.base_layer.global_num_experts,
            expert_map=self.base_layer.expert_map,
            custom_routing_function=self.base_layer.custom_routing_function,
            scoring_func=self.base_layer.scoring_func,
            routed_scaling_factor=self.base_layer.routed_scaling_factor,
            e_score_correction_bias=self.base_layer.e_score_correction_bias,
            activation=self.base_layer.activation,
            enable_eplb=self.base_layer.enable_eplb,
            expert_load_view=self.base_layer.expert_load_view,
            logical_to_physical_map=self.base_layer.logical_to_physical_map,
            logical_replica_count=self.base_layer.logical_replica_count,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        # Check if we have any active LoRA adapters
        active_adapters = self.get_active_adapters()
        
        if not active_adapters:
            # No active adapters, use the base layer's apply method
            return self.base_layer.apply(
                layer, x, router_logits, top_k, renormalize,
                use_grouped_topk, topk_group, num_expert_group,
                global_num_experts, expert_map, custom_routing_function,
                scoring_func, routed_scaling_factor, e_score_correction_bias,
                apply_router_weight_on_input, activation, enable_eplb,
                expert_load_view, logical_to_physical_map, logical_replica_count
            )

        # If we have active adapters, we must use the iterative implementation
        # because we need to inject LoRA computation per expert.
        # We assume VLLM_ENABLE_LORA_ON_MOE is effectively True here.
        
        # We need to gather LoRA weights for the active adapter
        # For simplicity, we support only one active adapter for now (the last one)
        # or we sum them up if multiple?
        # BaseLayerWithLoRA handles multiple adapters via `lora_a_stacked` etc.
        # But for MoE, the structure is complex.
        
        # Let's look at how `lora_a_stacked` is structured.
        # It is [max_loras, rank, input_dim].
        # But for MoE, we have multiple experts.
        # The LoRA weights for MoE experts should probably be:
        # [max_loras, num_experts, rank, input_dim]
        
        # However, vLLM's LoRA implementation usually stacks them.
        # If we registered `mlp.experts.0.down_proj`, `mlp.experts.1.down_proj`...
        # Then `FusedMoEWithLoRA` replaces `FusedMoE`.
        # `FusedMoE` contains ALL experts.
        # So `FusedMoEWithLoRA` should manage LoRA weights for ALL experts.
        
        # This is tricky because `BaseLayerWithLoRA` assumes a single weight matrix (or stacked for A/B).
        # Here we have `num_experts` weight matrices.
        
        # We need to override `create_lora_weights` to handle this?
        # Or we rely on the fact that `FusedMoE` is a single layer, so `target_modules`
        # should point to `mlp.experts`?
        # But `mlp.experts` is not a Linear layer.
        
        # If the user targets `mlp.experts.0.down_proj`, vLLM tries to replace `down_proj` of expert 0.
        # But `FusedMoE` does not have sub-modules for experts! It has `w13_weight` and `w2_weight` tensors.
        
        # So standard LoRA replacement fails because there are no `Linear` submodules to replace.
        
        # We need `FusedMoEWithLoRA` to hold the LoRA weights for all experts.
        # This means `FusedMoEWithLoRA` needs to be aware of which expert corresponds to which LoRA weight.
        
        # This requires significant changes to how LoRA weights are loaded and stored for FusedMoE.
        # Given the constraints, we might need a simpler hack.
        
        # Hack:
        # We can pass the `lora_dict` to `_moe_lora`?
        # But `FusedMoEWithLoRA` needs to have access to them.
        
        # Let's assume for now we can get the LoRA weights.
        # We will use `_moe_lora` (the python implementation) and pass the LoRA weights to it.
        # I will modify `_moe_lora` to accept `lora_a` and `lora_b` dicts/tensors.
        
        from vllm.model_executor.layers.fused_moe.moe_torch_iterative import fused_moe
        
        # We need to pass our instance to fused_moe or handle the logic here.
        # Let's handle the logic here by calling a modified _moe_lora directly.
        
        return self._moe_lora_forward(
            x, self.base_layer.w13_weight, self.base_layer.w2_weight,
            router_logits, top_k, global_num_experts, expert_map, renormalize
        )

    def _moe_lora_forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        global_num_experts: int,
        expert_map: torch.Tensor = None,
        renormalize: bool = False,
    ) -> torch.Tensor:
        
        # This is a modified version of _moe_lora from moe_torch_iterative.py
        # that applies LoRA.
        
        import torch.nn.functional as F
        
        orig_shape = hidden_states.shape
        hidden_size = hidden_states.shape[-1]
        num_tokens = hidden_states.shape[:-1].numel()
        num_experts = w1.shape[0]
        intermediate_size = w2.shape[-1]
        dtype = hidden_states.dtype
    
        hidden_states = hidden_states.view(num_tokens, hidden_size)
        gating_output = gating_output.view(num_tokens, global_num_experts)
        topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
        topk_weights, selected_experts = topk_weights.topk(topk, dim=-1)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)
    
        if expert_map is not None:
            selected_experts = expert_map[selected_experts]
    
        final_hidden_states = None
        
        # Get active adapter index
        active_adapters = self.get_active_adapters()
        if not active_adapters:
             # Should not happen given the check in apply()
             return torch.zeros_like(hidden_states)
             
        # For now, assume single active adapter
        # We use the last active adapter index
        adapter_index = active_adapters[-1]
        
        lora_a_list = self.lora_a_stacked[adapter_index]
        lora_b_list = self.lora_b_stacked[adapter_index]
        
        if lora_a_list is None or lora_b_list is None:
             return torch.zeros_like(hidden_states)

        for expert_idx in range(num_experts):
            expert_w1 = w1[expert_idx]
            expert_w2 = w2[expert_idx]
            expert_mask = selected_experts == expert_idx
            expert_weights = (topk_weights * expert_mask).sum(dim=-1, keepdim=True)
            
            # Standard computation
            x = F.linear(hidden_states, expert_w1)
            
            # Apply LoRA for W1 (Gate/Up)
            # Index in packed list: expert_idx * 2
            gate_up_idx = expert_idx * 2
            if gate_up_idx < len(lora_a_list):
                lora_a = lora_a_list[gate_up_idx]
                lora_b = lora_b_list[gate_up_idx]
                if lora_a is not None and lora_b is not None:
                    # lora_a: [rank, hidden]
                    # lora_b: [2 * intermediate, rank]
                    # delta = x @ A.T @ B.T
                    x += (hidden_states @ lora_a.T) @ lora_b.T
            
            gate = F.silu(x[:, :intermediate_size])
            x = x[:, intermediate_size:] * gate
            
            # Input to W2 is x
            input_to_w2 = x
            x = F.linear(x, expert_w2)
            
            # Apply LoRA for W2 (Down)
            # Index in packed list: expert_idx * 2 + 1
            down_idx = expert_idx * 2 + 1
            if down_idx < len(lora_a_list):
                lora_a = lora_a_list[down_idx]
                lora_b = lora_b_list[down_idx]
                if lora_a is not None and lora_b is not None:
                    # lora_a: [rank, intermediate]
                    # lora_b: [hidden, rank]
                    x += (input_to_w2 @ lora_a.T) @ lora_b.T
            
            current_hidden_states = x * expert_weights
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states = final_hidden_states + current_hidden_states
    
        return final_hidden_states.view(orig_shape)
