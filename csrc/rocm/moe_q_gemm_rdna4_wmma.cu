// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused MoE W4A16 GPTQ kernel for RDNA4 (gfx1200/gfx1201) using the gfx12
// 16x16x16 WMMA matrix cores.
//
// This is the prefill companion to moe_q_gemm_rdna3.cu. The scalar kernel
// wins at decode, where per-expert M is ~1-3 tokens; this kernel is used when
// a token block holds a full 16 rows (block_size_m == 16) and the 16x16x16
// WMMA tile is fully utilised.
//
// gfx12 WMMA 16x16x16 w32 fragment layout. Matches the AMD GPUOpen
// "WMMA guide for AMD RDNA 4 GPUs" sample (part 1, fused_gemm_TN) and the
// gfx12 staging in attention.cu:
//   A (M x K, 16-bit): lane16 = m, khalf = lane/16 = k / 8, element j = k % 8
//   B (K x N, 16-bit): lane16 = n, khalf = lane/16 = k / 8, element j = k % 8
//   D (M x N, fp32):   lane16 = n, mhalf = lane/16 = m / 8, element j = m % 8
// Both A and B are K-major: each lane holds 8 contiguous K elements (16 bytes
// for 16-bit data). The intrinsic computes D = op1 . op2, so passing the
// activation as op1 and the weight as op2 yields D[token][output].
//
// Weight format matches the scalar kernel: [E, K/8, N] uint32 shuffled,
// [E, groups, N] scales, [E, groups, N/8] packed zeros (GPTQv1, zero_offset 1).

#include <cstdint>
#include <type_traits>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "qdq_4_rdna3.cuh"

// gfx1200/gfx1201 only. gfx1250 is CDNA-classified and lacks these WMMA
// variants; the build never compiles this file for it.
#if defined(__HIPCC__) && (defined(__gfx1200__) || defined(__gfx1201__))
  #define __HIP__RDNA4_WMMA__
#endif

namespace vllm {
namespace moe_gptq_rdna4_wmma {

#if defined(__HIP__RDNA4_WMMA__) || !defined(__HIP_DEVICE_COMPILE__)

using bf16_t = __hip_bfloat16;
using floatx8 = float __attribute__((__vector_size__(8 * sizeof(float))));
using bit16x8 = uint16_t __attribute__((__vector_size__(8 * sizeof(uint16_t))));

template <typename T>
__device__ __forceinline__ floatx8 wmma_mma(bit16x8 a, bit16x8 b, floatx8 c) {
  if constexpr (std::is_same<T, bf16_t>::value) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
  } else {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
  }
}

// bf16 narrow without the defensive NaN canonicalisation hipcc emits for
// __float2bfloat16. Dequant outputs are bounded, never NaN/Inf.
__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) {
  uint32_t fu = __float_as_uint(f);
  uint32_t lsb = (fu >> 16) & 1u;
  return (uint16_t)((fu + 0x7FFFu + lsb) >> 16);
}

template <typename T>
__device__ __forceinline__ T to_T(float v) {
  if constexpr (std::is_same<T, bf16_t>::value) {
    return __float2bfloat16(v);
  } else {
    return __float2half(v);
  }
}

// Dequantize one shuffled int32 (8 nibbles) into the 8 B-fragment elements in
// natural k order. The shuffle places q[2k] at bits [4k:4k+3] and q[2k+1] at
// bits [16+4k:16+4k+3], so `(qa >> 4i) & 0x000F000F` selects the pair (q[2i],
// q[2i+1]) for i = 0..3. Result: element j = q[j], dequantized as
// scale * (q - zero).
template <typename T>
__device__ __forceinline__ bit16x8 dequant8(uint32_t qa, T scale,
                                            uint32_t zero) {
  bit16x8 out;
  uint16_t* o = (uint16_t*)&out;
  if constexpr (std::is_same<T, bf16_t>::value) {
    const float sf = __bfloat162float(scale);
    const float z = -(128.0f + (float)zero) * sf;
    const float y = sf;
    const uint32_t c0 = 0x43004300u;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      const uint32_t q = ((qa >> (4 * i)) & 0x000F000Fu) | c0;
      const float lo = __uint_as_float((q & 0xFFFFu) << 16);
      const float hi = __uint_as_float(q & 0xFFFF0000u);
      o[2 * i] = f32_to_bf16_bits(__fmaf_rn(lo, y, z));
      o[2 * i + 1] = f32_to_bf16_bits(__fmaf_rn(hi, y, z));
    }
  } else {
    const float sf = __half2float(scale);
    const half2 y = __half2half2(scale);
    const half2 z = __float2half2_rn(-(1024.0f + (float)zero) * sf);
    const uint32_t c0 = 0x64006400u;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      const uint32_t q = ((qa >> (4 * i)) & 0x000F000Fu) | c0;
      half2 dq;
      __builtin_memcpy(&dq, &q, sizeof(q));
      dq = __hfma2(dq, y, z);
      __builtin_memcpy(&o[2 * i], &dq, sizeof(dq));
    }
  }
  return out;
}

template <typename T>
__device__ __forceinline__ bit16x8 zero_frag() {
  bit16x8 z;
  uint16_t* o = (uint16_t*)&z;
  #pragma unroll
  for (int i = 0; i < 8; i++) o[i] = 0;
  return z;
}

// A fragment: lane16 = m, khalf = k/8, so this lane needs a[row][k .. k+7].
template <typename T>
__device__ __forceinline__ bit16x8 load_a(const T* a, int64_t row, int k,
                                          int size_k, bool valid) {
  if (!valid || k + 8 > size_k) return zero_frag<T>();
  bit16x8 out;
  __builtin_memcpy(&out, a + row * (int64_t)size_k + k, sizeof(out));
  return out;
}

// Element-wise atomic add for a 16-bit output. No native 16-bit CAS, so CAS
// the containing 32-bit word and preserve the sibling half. Only blocks that
// map to the same output row (top-k reduction) contend, and they write
// different n, so contention is low.
template <typename T>
__device__ __forceinline__ void atomic_add_out(T* addr, float v) {
  uint32_t* word = (uint32_t*)((uintptr_t)addr & ~(uintptr_t)3);
  const bool hi = (((uintptr_t)addr >> 1) & 1) != 0;
  uint32_t old = *word;
  while (true) {
    const uint16_t cur_bits =
        hi ? (uint16_t)(old >> 16) : (uint16_t)(old & 0xFFFFu);
    float f;
    if constexpr (std::is_same<T, bf16_t>::value) {
      bf16_t cur;
      __builtin_memcpy(&cur, &cur_bits, sizeof(cur));
      f = __bfloat162float(cur);
    } else {
      half cur;
      __builtin_memcpy(&cur, &cur_bits, sizeof(cur));
      f = __half2float(cur);
    }
    uint16_t sum_bits;
    if constexpr (std::is_same<T, bf16_t>::value) {
      bf16_t s = __float2bfloat16(f + v);
      __builtin_memcpy(&sum_bits, &s, sizeof(s));
    } else {
      half s = __float2half(f + v);
      __builtin_memcpy(&sum_bits, &s, sizeof(s));
    }
    const uint32_t nw = hi ? ((old & 0xFFFFu) | ((uint32_t)sum_bits << 16))
                           : ((old & 0xFFFF0000u) | (uint32_t)sum_bits);
    const uint32_t prev = atomicCAS(word, old, nw);
    if (prev == old) break;
    old = prev;
  }
}

// One block handles BLOCK_SIZE_M = 16 sorted tokens for one expert and
// NWARPS * 16 output columns. Each warp owns one 16x16 WMMA tile.
template <typename T, int NWARPS>
__global__ __launch_bounds__(NWARPS * 32) void moe_gemm_q4_wmma_kernel(
    const T* __restrict__ a, T* __restrict__ c,
    const uint32_t* __restrict__ b_q_weight,
    const T* __restrict__ b_scales,
    const uint32_t* __restrict__ b_qzeros,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_post_padded, int size_m, int size_n,
    int size_k, int groups, int top_k, int expert_weight_stride,
    int expert_scales_stride, int expert_zeros_stride, bool mul_topk_weight,
    int output_topk) {
  (void)num_tokens_post_padded;
  constexpr int BM = 16;
  constexpr int BN = 16;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int lane16 = lane & 15;
  const int khalf = lane >> 4;

  const int block_m = blockIdx.x;
  const int m_off = block_m * BM;
  const int expert = expert_ids[block_m];
  if (expert < 0) return;  // padding block or expert-parallel -1
  const int n = blockIdx.y * (NWARPS * BN) + warp * BN + lane16;
  const bool n_valid = n < size_n;
  const int groupsize = size_k / groups;

  // A fragment: this lane holds row m = lane16 at columns k0 + khalf*8 .. +7.
  const int token_id_a = sorted_token_ids[m_off + lane16];
  const int row_a = token_id_a / top_k;
  const bool row_a_valid = token_id_a >= 0 && row_a < size_m;

  const uint32_t* wptr = b_q_weight + (int64_t)expert * expert_weight_stride;
  const T* sptr = b_scales + (int64_t)expert * expert_scales_stride;
  const uint32_t* zptr = b_qzeros + (int64_t)expert * expert_zeros_stride;

  floatx8 acc;
  #pragma unroll
  for (int i = 0; i < 8; i++) acc[i] = 0.0f;

  for (int k0 = 0; k0 < size_k; k0 += 16) {
    const int k = k0 + khalf * 8;
    const bit16x8 afrag = load_a<T>(a, row_a, k, size_k, row_a_valid);

    bit16x8 bfrag = zero_frag<T>();
    if (n_valid) {
      // B fragment: this lane holds column n = lane16 at rows k0 + khalf*8.
      const uint32_t qa =
          wptr[(int64_t)(k0 / 8 + khalf) * size_n + n];
      const int g = k / groupsize;
      const T sc = sptr[(int64_t)g * size_n + n];
      const uint32_t zpacked = zptr[(int64_t)g * (size_n / 8) + (n >> 3)];
      const uint32_t zero = (zpacked >> (4 * (n & 7))) & 0xFu;
      bfrag = dequant8<T>(qa, sc, zero + 1u);
    }

    acc = wmma_mma<T>(afrag, bfrag, acc);
  }

  if (!n_valid) return;

  // C fragment: this lane holds column n = lane16 at rows m = khalf*8 .. +7.
  #pragma unroll
  for (int j = 0; j < 8; j++) {
    const int m = khalf * 8 + j;
    const int token_id = sorted_token_ids[m_off + m];
    if (token_id < 0 || token_id / top_k >= size_m) continue;

    float v = acc[j];
    if (mul_topk_weight && topk_weights != nullptr) {
      v *= topk_weights[token_id];
    }

    const int64_t out_row =
        (output_topk > 0) ? (int64_t)(token_id / output_topk)
                          : (int64_t)token_id;
    T* dst = c + out_row * size_n + n;
    if (output_topk > 0) {
      atomic_add_out<T>(dst, v);
    } else {
      *dst = to_T<T>(v);
    }
  }
}

#else  // non-RDNA4: empty stub for symbol parity

template <typename T, int NWARPS>
__global__ void moe_gemm_q4_wmma_kernel(
    const T*, T*, const uint32_t*, const T*, const uint32_t*, const float*,
    const int32_t*, const int32_t*, const int32_t*, int, int, int, int, int,
    int, int, int, bool, int) {}

#endif  // __HIP__RDNA4_WMMA__ || !__HIP_DEVICE_COMPILE__

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

template <typename T, int NWARPS>
void launch_moe_gemm_q4_wmma(
    const T* a, T* c, const uint32_t* b_q_weight, const T* b_scales,
    const uint32_t* b_qzeros, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded, int num_token_blocks, int size_m,
    int size_n, int size_k, int groups, int top_k, int expert_weight_stride,
    int expert_scales_stride, int expert_zeros_stride, bool mul_topk_weight,
    int output_topk, cudaStream_t stream) {
  dim3 block(NWARPS * 32);
  dim3 grid(num_token_blocks, (size_n + NWARPS * 16 - 1) / (NWARPS * 16), 1);
  moe_gemm_q4_wmma_kernel<T, NWARPS><<<grid, block, 0, stream>>>(
      a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
      expert_ids, num_tokens_post_padded, size_m, size_n, size_k, groups, top_k,
      expert_weight_stride, expert_scales_stride, expert_zeros_stride,
      mul_topk_weight, output_topk);
}

template <typename T>
void dispatch_moe_gemm_q4_wmma(
    const T* a, T* c, const uint32_t* b_q_weight, const T* b_scales,
    const uint32_t* b_qzeros, const float* topk_weights,
    const int32_t* sorted_token_ids, const int32_t* expert_ids,
    const int32_t* num_tokens_post_padded, int num_token_blocks, int size_m,
    int size_n, int size_k, int groups, int top_k, int expert_weight_stride,
    int expert_scales_stride, int expert_zeros_stride, bool mul_topk_weight,
    int output_topk, cudaStream_t stream) {
  launch_moe_gemm_q4_wmma<T, 4>(
      a, c, b_q_weight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
      expert_ids, num_tokens_post_padded, num_token_blocks, size_m, size_n,
      size_k, groups, top_k, expert_weight_stride, expert_scales_stride,
      expert_zeros_stride, mul_topk_weight, output_topk, stream);
}

}  // namespace moe_gptq_rdna4_wmma
}  // namespace vllm

// ---------------------------------------------------------------------------
// Public entry point. Signature mirrors moe_gptq_gemm_rdna3 so the Python
// experts can swap the op without reshaping buffers. block_size_m must be 16.
// ---------------------------------------------------------------------------

void moe_gptq_gemm_rdna4_wmma(torch::Tensor a, torch::Tensor c,
                              torch::Tensor b_q_weight, torch::Tensor b_scales,
                              torch::Tensor b_qzeros, torch::Tensor topk_weights,
                              torch::Tensor sorted_token_ids,
                              torch::Tensor expert_ids,
                              torch::Tensor num_tokens_post_padded,
                              int64_t top_k, int64_t block_size_m,
                              bool mul_topk_weight, int64_t output_topk) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA/HIP tensor");
  TORCH_CHECK(c.is_cuda(), "c must be a CUDA/HIP tensor");
  TORCH_CHECK(b_q_weight.is_cuda(), "b_q_weight must be a CUDA/HIP tensor");
  TORCH_CHECK(a.dim() == 2, "a must be 2D");
  TORCH_CHECK(c.dim() == 2, "c must be 2D");
  TORCH_CHECK(b_q_weight.dim() == 3, "b_q_weight must be 3D [E, K/8, N]");
  TORCH_CHECK(b_scales.dim() == 3, "b_scales must be 3D [E, groups, N]");
  TORCH_CHECK(b_qzeros.dim() == 3, "b_qzeros must be 3D [E, groups, N/8]");
  TORCH_CHECK(block_size_m == 16,
              "moe_gptq_gemm_rdna4_wmma requires block_size_m == 16, got ",
              block_size_m);
  TORCH_CHECK(
      a.scalar_type() == torch::kHalf || a.scalar_type() == torch::kBFloat16,
      "a must be half or bfloat16");
  TORCH_CHECK(a.scalar_type() == b_scales.scalar_type(),
              "b_scales dtype must match a");
  TORCH_CHECK(b_q_weight.size(2) == c.size(1),
              "c must have b_q_weight's N as its last dim");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto stream = at::cuda::getCurrentCUDAStream();

  int size_m = (int)a.size(0);
  int size_k = (int)a.size(1);
  int size_n = (int)b_q_weight.size(2);
  int groups = (int)b_scales.size(1);

  TORCH_CHECK(size_k % 16 == 0, "size_k must be a multiple of 16, got ", size_k);
  TORCH_CHECK(size_n % 8 == 0, "size_n must be a multiple of 8, got ", size_n);
  TORCH_CHECK(groups > 0 && size_k % groups == 0 && size_k / groups >= 8,
              "group_size must divide size_k and be >= 8");

  int expert_weight_stride = (int)(b_q_weight.size(1) * b_q_weight.size(2));
  int expert_scales_stride = (int)(b_scales.size(1) * b_scales.size(2));
  int expert_zeros_stride = (int)(b_qzeros.size(1) * b_qzeros.size(2));

  int num_token_blocks = (int)(sorted_token_ids.size(0) / block_size_m);

  const float* topk_w_ptr =
      (topk_weights.numel() > 0) ? topk_weights.data_ptr<float>() : nullptr;

  using vllm::gptq_rdna3::bf16_t;

  auto dispatch = [&](auto* a_ptr, auto* c_ptr, const auto* s_ptr) {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(a_ptr)>>;
    vllm::moe_gptq_rdna4_wmma::dispatch_moe_gemm_q4_wmma<T>(
        a_ptr, c_ptr, (const uint32_t*)b_q_weight.data_ptr<int32_t>(), s_ptr,
        (const uint32_t*)b_qzeros.data_ptr<int32_t>(), topk_w_ptr,
        sorted_token_ids.data_ptr<int32_t>(), expert_ids.data_ptr<int32_t>(),
        num_tokens_post_padded.data_ptr<int32_t>(), num_token_blocks, size_m,
        size_n, size_k, groups, (int)top_k, expert_weight_stride,
        expert_scales_stride, expert_zeros_stride, mul_topk_weight,
        (int)output_topk, stream);
  };

  if (a.scalar_type() == torch::kHalf) {
    dispatch((const half*)a.data_ptr(), (half*)c.data_ptr(),
             (const half*)b_scales.data_ptr());
  } else {
    dispatch((const bf16_t*)a.data_ptr(), (bf16_t*)c.data_ptr(),
             (const bf16_t*)b_scales.data_ptr());
  }
}