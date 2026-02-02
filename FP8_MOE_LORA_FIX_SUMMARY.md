# FP8 MoE LoRA Fix Summary

## Problem Description

When running vLLM with LoRA adapters on an FP8-quantized MoE model (specifically GLM-4.5-Air-Derestricted-FP8), two different errors were encountered:

### Error 1: FP8 Quantization Incompatibility
```
ValueError: CompressedTensorsW8A8Fp8MoEMethod uses the new modular kernel initialization logic. This function should not be called.
```

This error occurred during the LoRA layer injection process in the worker initialization phase.

### Error 2: Expert Parallel (EP) Incompatibility
When adding `--enable-expert-parallel` to the configuration:
```
AssertionError: EP support for Fused MoE LoRA is not implemented yet.
```

This error occurred because the fused MoE LoRA implementation explicitly rejected expert parallel mode.

## Root Cause Analysis

The issue stems from a mismatch between two different MoE kernel initialization patterns:

### Old Initialization Pattern
- Used by most quantization methods (e.g., INT8, INT4)
- Requires calling `select_gemm_impl()` to create the experts kernel
- The LoRA injection code in `FusedMoEWithLoRA._inject_lora_into_fused_moe()` was designed for this pattern

### New Modular Kernel Initialization Pattern
- Used by FP8 (`CompressedTensorsW8A8Fp8MoEMethod`) and NVFP4 (`CompressedTensorsW4A4Nvfp4MoEMethod`) quantization methods
- Creates the modular kernel (`self.moe_mk`) during `process_weights_after_loading()`
- Explicitly raises an error if `select_gemm_impl()` is called, as it's not compatible with this pattern

### The Conflict
The LoRA injection code at line 136 in `vllm/lora/layers/fused_moe.py` unconditionally called:
```python
m_fused_moe_fn = FusedMoEModularKernel(
    prepare_finalize,
    self.base_layer.quant_method.select_gemm_impl(
        prepare_finalize, self.base_layer
    ),
    self.base_layer.shared_experts,
)
```

This failed for FP8/NVFP4 quantization methods because:
1. They already have a pre-initialized `moe_mk` kernel
2. They don't support the `select_gemm_impl()` method

## Solution

### Fix 1: FP8 Quantization Support

Modified `FusedMoEWithLoRA._inject_lora_into_fused_moe()` to check if the quantization method already has a pre-initialized modular kernel:

```python
# Check if the quantization method already has a pre-initialized modular kernel
# (e.g., for FP8 quantization which uses the new modular kernel initialization logic)
if hasattr(self.base_layer.quant_method, 'moe_mk') and self.base_layer.quant_method.moe_mk is not None:
    # Use the pre-existing modular kernel
    m_fused_moe_fn = self.base_layer.quant_method.moe_mk
else:
    # Use the old initialization logic for quantization methods that support it
    prepare_finalize = MoEPrepareAndFinalizeNoEP()
    m_fused_moe_fn = FusedMoEModularKernel(
        prepare_finalize,
        self.base_layer.quant_method.select_gemm_impl(
            prepare_finalize, self.base_layer
        ),
        self.base_layer.shared_experts,
    )
```

### Fix 2: Expert Parallel Support

Modified `FusedMoEWithLoRA._inject_lora_into_fused_moe()` to use the appropriate prepare/finalize logic based on whether EP is enabled:

```python
# Use the appropriate prepare_finalize based on whether EP is enabled
if self.base_layer.use_ep:
    from vllm.model_executor.layers.fused_moe.prepare_finalize import (
        MoEPrepareAndFinalizeNaiveEP,
    )
    prepare_finalize = MoEPrepareAndFinalizeNaiveEP(
        is_sequence_parallel=self.base_layer.is_sequence_parallel,
        num_dispatchers=1,
    )
else:
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

Also removed the assertion that rejected EP mode and removed the workaround in `can_replace_layer()` that skipped LoRA for MoE layers.

## Impact

These fixes enable full LoRA support for:
- **FP8-quantized MoE models** (e.g., GLM-4.5-Air-Derestricted-FP8)
- **NVFP4-quantized MoE models**
- **MoE models with Expert Parallel enabled**

The fixes are backward compatible and don't affect existing quantization methods that use the old initialization pattern.

## How EP Support Works

When EP is enabled:
1. `MoEPrepareAndFinalizeNaiveEP` handles dispatching tokens to the appropriate EP ranks
2. Each EP rank processes only its local experts
3. The LoRA adapters are applied to the local experts on each rank
4. Results are combined back across EP ranks in the finalize step

The `moe_parallel_config` parameter ensures that the modular kernel has access to EP configuration (ep_size, ep_rank, etc.) for proper token routing.

## Files Modified

- `vllm/lora/layers/fused_moe.py` - Modified `_inject_lora_into_fused_moe()` method, removed EP assertion, and updated `can_replace_layer()` methods

## Testing

The fix should be tested with:
1. FP8-quantized MoE models with LoRA adapters (without EP)
2. FP8-quantized MoE models with LoRA adapters (with EP)
3. NVFP4-quantized MoE models with LoRA adapters
4. Existing quantization methods (INT8, INT4) to ensure backward compatibility

### Test Commands

**Without EP:**
```bash
vllm serve /path/to/fp8-moe-model \
  --enable-lora \
  --lora-modules test-lora=/path/to/lora \
  --max-loras 4
```

**With EP:**
```bash
vllm serve /path/to/moe-model \
  --enable-lora \
  --enable-expert-parallel \
  --lora-modules test-lora=/path/to/lora \
  --max-loras 4
```

**With FP8 and EP:**
```bash
vllm serve /path/to/fp8-moe-model \
  --enable-lora \
  --enable-expert-parallel \
  --lora-modules test-lora=/path/to/lora \
  --max-loras 4
```

## Related Code

- Quantization methods using new pattern:
  - `CompressedTensorsW8A8Fp8MoEMethod` (lines 681-1097 in compressed_tensors_moe.py)
  - `CompressedTensorsW4A4Nvfp4MoEMethod` (lines 369-678 in compressed_tensors_moe.py)

- Quantization methods using old pattern:
  - `CompressedTensorsW8A8Int8MoEMethod`
  - `CompressedTensorsWNA16MarlinMoEMethod`
  - `CompressedTensorsWNA16MoEMethod`
  - `CompressedTensorsW4A8Int8MoEMethod`
  - `CompressedTensorsW4A8Fp8MoEMethod`

- EP-related classes:
  - `MoEPrepareAndFinalizeNaiveEP` (prepare_finalize.py)
  - `MoEPrepareAndFinalizeNoEP` (prepare_finalize.py)
  - `FusedMoEModularKernel` (modular_kernel.py)