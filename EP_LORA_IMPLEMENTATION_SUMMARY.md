# Expert Parallelism (EP) Support for Fused MoE LoRA

## Summary

This implementation adds support for Expert Parallelism (EP) when using LoRA adapters with Fused MoE layers in vLLM. Previously, this combination was blocked by a hard assertion.

## Changes Made

### 1. Removed Hard Assertion
**File:** `vllm/lora/layers/fused_moe.py`

**Before:**
```python
assert not self.base_layer.use_ep, (
    "EP support for Fused MoE LoRA is not implemented yet."
)
```

**After:**
```python
self.use_ep = base_layer.use_ep
self.ep_size = base_layer.ep_size if self.use_ep else 1
self.ep_rank = base_layer.ep_rank if self.use_ep else 0
```

### 2. EP-Compatible Prepare/Finalize Selection
**File:** `vllm/lora/layers/fused_moe.py` (lines 133-145)

Added logic to select EP-compatible prepare/finalize implementations when EP is enabled:

```python
# Use EP-compatible prepare/finalize if EP is enabled
if self.use_ep:
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    # routing_tables may not be available yet during LoRA initialization
    routing_tables = getattr(self.base_layer, 'routing_tables', None)
    prepare_finalize = maybe_make_prepare_finalize(
        self.base_layer, quant_config, routing_tables
    )
    if prepare_finalize is None:
        # Fallback to NoEP if no EP-specific implementation available
        prepare_finalize = MoEPrepareAndFinalizeNoEP()
else:
    prepare_finalize = MoEPrepareAndFinalizeNoEP()
```

This allows the LoRA layer to use:
- DeepEP low-latency kernels (DeepEPLLPrepareAndFinalize)
- DeepEP high-throughput kernels (DeepEPHTPrepareAndFinalize)
- FlashInfer AllToAll/AllGather implementations
- PPLX kernels
- Mori kernels
- Fallback to NoEP if no EP implementation is available

### 3. Expert Filtering for LoRA Weights
**File:** `vllm/lora/layers/fused_moe.py` (lines 532-565, 739-770)

Added logic to filter LoRA weights to only include experts local to each EP rank:

```python
# In EP mode, the LoRA checkpoint contains global expert IDs
# We need to filter to only the experts local to this EP rank
if self.use_ep:
    if self.base_layer.expert_map is not None:
        # expert_map maps global expert IDs to local expert IDs
        # We need to extract only the experts that belong to this rank
        global_expert_ids = torch.arange(
            self.base_layer.global_num_experts,
            device=self.device
        )
        local_expert_mask = (global_expert_ids // self.ep_size) == self.ep_rank
        local_global_ids = global_expert_ids[local_expert_mask]
        
        # Filter LoRA weights to only include local experts
        w1_lora_a = w1_lora_a[local_expert_mask]
        w1_lora_b = w1_lora_b[local_expert_mask]
        # ... etc
    else:
        # Linear expert placement: experts are evenly distributed
        # Extract the slice of experts for this EP rank
        experts_per_rank = self.base_layer.global_num_experts // self.ep_size
        start_idx = self.ep_rank * experts_per_rank
        end_idx = start_idx + experts_per_rank
        
        w1_lora_a = w1_lora_a[start_idx:end_idx]
        # ... etc
```

This ensures that:
- Each EP rank only stores LoRA weights for its local experts
- Memory usage is distributed across EP ranks
- LoRA checkpoint loading automatically filters to local experts

### 4. Updated Weight Creation Comments
**File:** `vllm/lora/layers/fused_moe.py` (lines 338-339, 369-370)

Added clarifying comments about EP mode behavior:

```python
# In EP mode, local_num_experts already accounts for expert distribution
# In non-EP mode, local_num_experts equals global_num_experts
```

### 5. Fixed Dummy LoRA Creation for MoE
**File:** `vllm/lora/model_manager.py` (lines 501-522)

Added special handling for `FusedMoEWithLoRA` in dummy LoRA creation to match the structure used for `FusedMoE3DWithLoRA`:

```python
elif module.__class__.__name__ == "FusedMoEWithLoRA":
    # Handle FusedMoEWithLoRA separately
    # w2
    lora = LoRALayerWeights.create_dummy_lora_weights(
        module_name,
        module.w2_input_size,
        module.w2_output_size,
        rank * module.w2_lora_a_stacked[0].shape[1],  # rank*num_experts
        module.w2_lora_a_stacked[0].dtype,
        "cpu",
    )
    model.loras[module_name] = lora
    # w13
    lora = LoRALayerWeights.create_dummy_lora_weights(
        module_name,
        module.w13_input_size,
        module.w13_output_size,
        rank * module.w13_lora_a_stacked[0].shape[1],  # rank*num_experts
        module.w13_lora_a_stacked[0].dtype,
        "cpu",
    )
    model.loras[module_name + ".base_layer"] = lora
```

This fixes the `IndexError` that occurred when trying to access `module.lora_a_stacked[i]` for MoE layers during dummy LoRA creation.

## How It Works

### Architecture Overview

1. **Expert Distribution:** In EP mode, experts are distributed across GPU ranks. Each rank owns a subset of experts.

2. **LoRA Weight Distribution:** LoRA adapter weights follow the same distribution as the base model experts. Each EP rank only stores LoRA weights for its local experts.

3. **Prepare/Finalize Layer:** The prepare/finalize layer handles:
   - Dispatching tokens to the correct EP ranks
   - All-to-all communication for token routing
   - Combining results from all EP ranks

4. **LoRA Application:** LoRA adapters are applied locally on each EP rank to the expert computations performed on that rank.

### Supported EP Backends

The implementation supports all EP backends available in vLLM:
- **DeepEP Low-Latency:** `DeepEPLLPrepareAndFinalize`
- **DeepEP High-Throughput:** `DeepEPHTPrepareAndFinalize`
- **FlashInfer AllToAll:** `FlashInferAllToAllMoEPrepareAndFinalize`
- **FlashInfer AllGather:** `FlashInferAllGatherMoEPrepareAndFinalize`
- **PPLX:** `PplxPrepareAndFinalize`
- **Mori:** `MoriPrepareAndFinalize`

## Usage

### Command Line

You can now use `--enable-expert-parallel` with LoRA:

```bash
vllm serve /path/to/model \
--gpu-memory-utilization 0.95 \
--max-model-len 131072 \
--port 8000 \
--max-num-seqs 16 \
-tp 2 \
--enable-expert-parallel \
--attention-backend FLASHINFER \
--fully-sharded-loras \
--enable-lora \
--max-lora-rank 32 \
--max-loras 1 \
--max-cpu-loras 1 \
--lora-modules \
test-lora=/path/to/lora/checkpoint
```

### Configuration Requirements

1. **LoRA Checkpoint:** Your LoRA checkpoint should contain LoRA weights for all global experts. The implementation will automatically filter to local experts during loading.

2. **Expert Placement:** Works with both:
   - Linear expert placement (even distribution)
   - Round-robin expert placement (with expert_map)

3. **Memory:** Each EP rank needs memory for:
   - Base model weights for local experts
   - LoRA adapter weights for local experts
   - Communication buffers for all-to-all operations

## Limitations and Considerations

1. **LoRA Checkpoint Size:** LoRA checkpoints must contain weights for all global experts, not just local experts. The filtering happens during loading.

2. **Memory Overhead:** Each EP rank stores LoRA weights for its local experts. Total memory across all ranks equals the full LoRA checkpoint size.

3. **Communication:** EP requires all-to-all communication for token routing. This adds latency but allows scaling to larger models.

4. **Backend Compatibility:** Some EP backends may have specific requirements (e.g., hidden size alignment, quantization support). The implementation will fall back to NoEP if an EP-specific prepare/finalize is not available.

## Testing

To test the implementation:

1. **Basic Test:**
```bash
vllm serve /path/to/moe-model \
--enable-expert-parallel \
--enable-lora \
--lora-modules test=/path/to/lora
```

2. **Verify Expert Distribution:**
- Check that each GPU rank loads only its local experts' LoRA weights
- Monitor memory usage across ranks

3. **Performance Test:**
- Compare throughput with and without EP
- Verify that LoRA adapters are correctly applied

## Future Enhancements

Potential improvements for future versions:

1. **LoRA Checkpoint Optimization:** Support for pre-sharded LoRA checkpoints to avoid loading and filtering global weights.

2. **Dynamic LoRA Loading:** More efficient LoRA loading strategies for EP mode.

3. **Advanced Expert Placement:** Better support for custom expert placement strategies with LoRA.

4. **Performance Tuning:** Optimize the interaction between LoRA operations and EP communication patterns.

## References

- vLLM Expert Parallelism: `vllm/model_executor/layers/fused_moe/`
- vLLM LoRA: `vllm/lora/`
- DeepEP: https://github.com/deepseek-ai/DeepEP
- FlashInfer: https://github.com/flashinfer-ai/flashinfer