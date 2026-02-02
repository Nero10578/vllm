# Complete EP+LoRA Fix Summary

## Problem Description

The user was experiencing multiple errors when trying to run vLLM with LoRA adapters on an FP8-quantized MoE model (GLM-4.5-Air-Derestricted-FP8) with Expert Parallel (EP) enabled:

1. **FP8 Quantization Error**: `ValueError: CompressedTensorsW8A8Fp8MoEMethod uses the new modular kernel initialization logic. This function should not be called.`
2. **EP Assertion Error**: `AssertionError: EP support for Fused MoE LoRA is not implemented yet.`
3. **EP Weight Sharding Error**: Tensor shape mismatches during LoRA weight loading
4. **CUDA Graph Capture Crashes**: Illegal memory access errors during model initialization

## Root Cause Analysis

The issues stemmed from incompatibilities between:
- FP8 quantization's new modular kernel initialization pattern
- LoRA injection code designed for the old kernel initialization pattern
- Expert Parallel (EP) sharding of MoE experts across GPUs
- LoRA weight tensor storage and copying logic

## Complete Fix Implementation

### Fix 1: EP-Aware Kernel Initialization (Critical Fix)

**File**: `vllm/lora/layers/fused_moe.py`
**Method**: `_inject_lora_into_fused_moe()`

**Problem**: When EP is enabled, FP8 quantization methods were using pre-initialized kernels that don't support EP, leading to CUDA illegal memory access errors during graph capture.

**Solution**: Reordered the conditional logic to prioritize EP-aware kernel initialization when EP is enabled, regardless of whether a pre-initialized kernel exists:

```python
# When EP is enabled, we must use EP-aware kernel initialization
# even if the quantization method has a pre-initialized kernel
if self.base_layer.use_ep:
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNaiveEP,
    )
    prepare_finalize = MoEPrepareAndFinalizeNaiveEP(
        is_sequence_parallel=self.base_layer.is_sequence_parallel,
        num_dispatchers=1,
    )
    
    m_fused_moe_fn = FusedMoEModularKernel(
        prepare_finalize,
        self.base_layer.quant_method.select_gemm_impl(
            prepare_finalize, self.base_layer
        ),
        self.base_layer.shared_experts,
        moe_parallel_config=self.base_layer.moe_parallel_config,
    )
else:
    # For non-EP case, check if the quantization method already has a pre-initialized modular kernel
    # (e.g., for FP8 quantization which uses the new modular kernel initialization logic)
    if hasattr(self.base_layer.quant_method, 'moe_mk') and self.base_layer.quant_method.moe_mk is not None:
        # Use the pre-existing modular kernel
        m_fused_moe_fn = self.base_layer.quant_method.moe_mk
    else:
        # Use the standard NoEP prepare_finalize
        prepare_finalize = MoEPrepareAndFinalizeNoEP()
        
        m_fused_moe_fn = FusedMoEModularKernel(
            prepare_finalize,
            self.base_layer.quant_method.select_gemm_impl(
                prepare_finalize, self.base_layer
            ),
            self.base_layer.shared_experts,
            moe_parallel_config=self.base_layer.moe_parallel_config,
        )
```

**Key Insight**: EP compatibility must take precedence over quantization method optimizations. The pre-initialized FP8 kernels are not EP-aware and cause memory access violations when used with EP.

### Fix 2: FP8 Quantization Compatibility (Non-EP Case)

**Problem**: For non-EP cases, LoRA injection code unconditionally called `select_gemm_impl()`, which FP8/NVFP4 quantization methods don't support.

**Solution**: Maintained backward compatibility by using pre-initialized `moe_mk` kernel for non-EP FP8 quantization cases.

### Fix 3: EP-Aware LoRA Weight Sharding

**Problem**: LoRA weights were loaded for all global experts but each EP rank only has local experts.

**Solution**: Modified `set_lora()` methods in both `FusedMoEWithLoRA` and `FusedMoE3DWithLoRA` to slice LoRA weights based on EP rank using the expert_map:

```python
# When EP is enabled, slice LoRA weights to only include local experts
if self.ep_size > 1:
    expert_map = self.base_layer.expert_map
    if expert_map is not None:
        # expert_map maps global expert indices to local expert indices
        # -1 means the expert is not on this rank
        # We need to filter to only include experts that are local to this rank
        local_expert_indices = []
        for global_expert_idx in range(w1_lora_a.shape[0]):
            local_expert_idx = expert_map[global_expert_idx].item()
            if local_expert_idx != -1:
                local_expert_indices.append(global_expert_idx)
        
        # Slice to only include local experts
        if local_expert_indices:
            w1_lora_a = w1_lora_a[local_expert_indices]
            w2_lora_a = w2_lora_a[local_expert_indices]
            w3_lora_a = w3_lora_a[local_expert_indices]
            w1_lora_b = w1_lora_b[local_expert_indices]
            w2_lora_b = w2_lora_b[local_expert_indices]
            w3_lora_b = w3_lora_b[local_expert_indices]
        else:
            # No local experts, create empty tensors
            w1_lora_a = torch.zeros(0, *w1_lora_a.shape[1:], dtype=w1_lora_a.dtype, device=w1_lora_a.device)
            # ... similar for other tensors
```

### Fix 4: EP-Safe Tensor Creation

**Problem**: LoRA weight tensors were created with incorrect dimensions that didn't account for EP sharding.

**Solution**: Modified `_create_lora_a_weights()` and `_create_lora_b_weights()` to use the correct number of experts for the current EP rank:

```python
def _create_lora_a_weights(self, max_loras: int, lora_config: LoRAConfig):
    # When EP is enabled, we need to account for the actual number of experts
    # that will be stored on this rank after EP sharding
    num_experts = self.base_layer.local_num_experts
    
    self.w13_lora_a_stacked: tuple[torch.Tensor, ...] = tuple(
        torch.zeros(
            (
                max_loras,
                num_experts,  # Use the correct number of experts for this rank
                lora_config.max_lora_rank
                if not self.fully_sharded
                else divide(lora_config.max_lora_rank, self.tp_size),
                self.base_layer.hidden_size,
            ),
            dtype=lora_config.lora_dtype,
            device=self.device,
        )
        for _ in range(self._w13_slices)
    )
    # ... similar for w2_lora_a_stacked
```

### Fix 5: Simplified Tensor Copying

**Problem**: Complex tensor copying logic with dimension checks was causing CUDA graph capture crashes.

**Solution**: Simplified the tensor copying logic to rely on the correct tensor dimensions established during creation:

```python
# Copy the LoRA weights - the dimensions should now match correctly
# because we've properly handled EP sharding in the tensor creation
self.w13_lora_a_stacked[0][
    index, :, : slliced_w1_lora_a.shape[1], : slliced_w1_lora_a.shape[2]
].copy_(slliced_w1_lora_a, non_blocking=True)
# ... similar for other tensors
```

### Fix 6: Import Error Resolution

**Problem**: Code was trying to import non-existent functions `get_ep_rank` and `get_ep_size`.

**Solution**: Modified imports to use `get_ep_group()` and extract EP size/rank from the group object:

```python
from vllm.distributed.parallel_state import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

# In __init__:
try:
    ep_group = get_ep_group()
    self.ep_size = ep_group.world_size
    self.ep_rank = ep_group.rank_in_group
except AssertionError:
    # EP group not initialized (non-MoE model or EP not enabled)
    self.ep_size = 1
    self.ep_rank = 0
```

## Key Technical Concepts

- **FP8 Quantization**: 8-bit floating point quantization using `CompressedTensorsW8A8Fp8MoEMethod`
- **Modular Kernel Initialization**: New pattern where quantization methods pre-initialize kernels during `process_weights_after_loading()`
- **Expert Parallel (EP)**: Parallelization technique for distributing MoE experts across GPUs
- **Expert Map**: Tensor mapping global expert indices to local expert indices (-1 means expert not on this rank)
- **Fused MoE**: Optimized MoE implementation that fuses multiple operations
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning technique
- **Tensor Parallel (TP)**: Parallelization technique for distributing model layers across GPUs

## Testing Command

The fixes enable the following command to work correctly:

```bash
vllm serve /home/arli/models/GLM-4.5-Air-Derestricted-FP8 \
--gpu-memory-utilization 0.95 --max-model-len 131072 --port 8000 \
--max-num-seqs 16 -tp 2 \
--tool-call-parser glm45 --reasoning-parser glm45 --enable-auto-tool-choice \
--served-model-name GLM-4.5-Air \
--fully-sharded-loras --enable-lora --max-lora-rank 32 --max-loras 4 --max-cpu-loras 4 --lora-modules \
test-lora-1=/home/arli/train/glm45a-d/outputs/lora-out-derestricted-tuned/checkpoint-130 \
test-lora-2=/home/arli/train/glm45a-d/outputs/lora-out-derestricted-tuned-try2/checkpoint-130 \
test-lora-3=/home/arli/train/glm45a-d/outputs/lora-out-derestricted-tuned-try3/checkpoint-130 \
test-lora-4=/home/arli/train/glm45a-d/outputs/lora-out-derestricted-tuned-try4/checkpoint-130 \
--enable-expert-parallel
```

## Summary

This comprehensive fix addresses all the incompatibilities between FP8 quantization, Expert Parallel, and LoRA systems in vLLM. The solution:

1. **Maintains compatibility** with both old and new quantization kernel initialization patterns
2. **Properly handles EP sharding** by filtering LoRA weights to local experts only
3. **Ensures tensor dimension consistency** throughout the LoRA weight lifecycle
4. **Enables full LoRA support** for MoE layers even when EP is enabled

The model should now start successfully with LoRA adapters applied to both MoE and non-MoE layers, with proper expert parallel distribution across GPUs.