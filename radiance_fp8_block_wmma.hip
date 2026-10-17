// RADIANCE gfx1201 dense block-FP8 GEMM: C[M,N] bf16 = (A * As) . (W * Bs)^T
//
//   A   [M,K]        e4m3, row-major activation
//   W   [N,K]        e4m3, row-major weight (compressed-tensors FP8 block)
//   As  [M, K/128]   fp32, activation scales, group shape (1,128)
//   Bs  [N/128,K/128] fp32, weight scales,  block shape (128,128)
//   C   [M,N]        bf16
//
// Why: vLLM's ROCm block-FP8 route is the Triton kernel, which will not emit gfx1201's
// v_wmma_f32_16x16x16_fp8_fp8 -- it upconverts to 16-bit WMMA. The tuned Triton config runs at
// 139-159 TFLOP/s, ~87-99% of the 160.2 bf16 WMMA ceiling, so it has no headroom; this kernel is
// the same math on the fp8 instruction.
//
// Geometry and structure are copied from the production gfx1201 block-FP8 GEMM in the radiance
// engine (libr4d/r4d_gemm_fp8a8.hip, `gemm_fp8a8_tiled`), whose measured numbers are the reference
// this kernel is chasing (docs/KERNELS.md: 150-162 TFLOP/s at M=512). What was adopted, and why:
//
//   * 128x128 block face, 8 waves, one m-fragment and eight n-fragments per wave. "1x4 (FN = 64,
//     four resident blocks instead of three): ~20% slower"; "MT*NT > 8 spills". Eight accumulator
//     tiles per wave is the tuned point.
//   * FN = 128 = the scale block, so the weight scale folds as ONE scalar for the whole block and
//     all eight n-fragments share one A fragment set.
//   * The fold runs on BOTH accumulator sets (`raw` -> `out`) and fires once per 128 of K. The test
//     stays a runtime branch: gating it at compile-time-known positions cuts the instruction count
//     but raises the kernel's VGPRs and is ~16% slower, because the fold's live range, not its
//     instruction count, is the cost.
//   * Double-buffered staging, so a K tile needs one publish __syncthreads instead of two (~5%).
//   * The padded LDS pitch FK+8 is deliberate: 18r % 32 over r = 0..15 is sixteen distinct even
//     banks, one bank cycle per fragment read. An XOR swizzle cannot do better (rows r and r+8
//     still collide), and a +16 pad collides again.
//   * No grouped-M CTA swizzle. The weight re-read per M tile is absorbed by the 64 MB MALL, which
//     is why the swizzle measured as a no-op here.
//   * No fragment prefetch across k-steps: the fragment reads are not the limit (measured: a grid
//     with a third fewer fragment reads ties).
//   * Waves per SIMD pinned (RAD_WAVES(6)), as libr4d does, so the compiler cannot trade a load
//     round trip per WMMA for one more wave.
//
// Numerics: identical semantics to the Triton kernel -- fp32 accumulation, the same per-(1,128)
// activation scale and per-(128,128) weight scale applied inside the K loop. Accumulation order
// differs, so it is gated by a relative-error check and by RADIANCE_FP8_WMMA.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
#include <cstdlib>
namespace py = pybind11;

typedef float floatx8 __attribute__((ext_vector_type(8)));   // f32 accumulator, 8 VGPRs
typedef int i2v __attribute__((ext_vector_type(2)));         // 8-byte fp8 fragment
typedef unsigned int uint2_t __attribute__((ext_vector_type(2)));

// Pin the occupancy the kernel is scheduled for, as libr4d does (R4D_WAVES). Left to itself the
// scheduler's target moves between compiler releases -- LLVM 24 serialises a GEMM's loads to win
// one more wave, paying a load round trip per WMMA. 6 is the value the reference prefill tile uses,
// and this kernel now has the reference's register budget (8 accumulator tiles x 2 sets).
#if defined(__has_attribute)
#  if __has_attribute(amdgpu_waves_per_eu)
#    define RAD_WAVES(n) __attribute__((amdgpu_waves_per_eu((n), (n))))
#  else
#    define RAD_WAVES(n)
#  endif
#else
#  define RAD_WAVES(n)
#endif

#define BM 128            // M rows per block
#define BN 128            // N rows per block; equals the weight scale block
#define BK 64             // K per staged tile, in bytes (one LDS row)
#define KBLK 128          // scale block along K, on both operands
#define NWAVE 8
#define NTHREADS (NWAVE * 32)
#define NT 8              // n-fragments (and so accumulator tiles) per wave
#define ASTR (BK + 8)     // padded LDS pitch: see the bank note above
#define WSTR (BK + 8)

// One 16x16 fragment: lane l holds row (l & 15), k at k0 + (l >> 4) * 8, eight bytes.
__device__ __forceinline__ i2v frag(const unsigned char *__restrict__ lds, int stride, int row0,
                                    int k0, int lane) {
  const unsigned char *p = &lds[(row0 + (lane & 15)) * stride + k0 + ((lane >> 4) * 8)];
  i2v r;
  r[0] = *(const int *)p;
  r[1] = *(const int *)(p + 4);
  return r;
}

// 32 contiguous bytes a thread stages, as four 8-byte stores so the padded pitch stays aligned.
__device__ __forceinline__ void stage_store(unsigned char *__restrict__ row, const uint2_t v[4]) {
#pragma unroll
  for (int g = 0; g < 4; ++g) *(uint2_t *)(row + 8 * g) = v[g];
}
__device__ __forceinline__ void stage_load(uint2_t v[4], const unsigned char *__restrict__ src) {
#pragma unroll
  for (int g = 0; g < 4; ++g) v[g] = *(const uint2_t *)(src + 8 * g);
}

__global__ __launch_bounds__(NTHREADS) RAD_WAVES(6) void radiance_fp8_block_wmma_gemm(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const float *__restrict__ As, const float *__restrict__ Bs,
    __bf16 *__restrict__ C, int M, int N, int K) {
  __shared__ unsigned char s_a[2][BM * ASTR];
  __shared__ unsigned char s_b[2][BN * WSTR];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;   // grid runs x-fastest over N
  const int kgrp = K / KBLK;
  const int nb = n0 / BN;                                 // FN == BN, so one weight scale per block
  // This wave's A rows, and its column for the output.
  const int wrow = wave * 16;

  // Staging assignment: two threads per row, 32 B each, so a pass covers all 128 rows.
  const int sn = tid >> 1, sk = (tid & 1) * 32;
  const int ra_c = sn < M - 1 - m0 ? sn : M - 1 - m0;

  floatx8 raw[NT], out[NT];
#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) { raw[j][e] = 0.f; out[j][e] = 0.f; }

  uint2_t pa[4], pb[4];
  stage_load(pa, A + (size_t)(m0 + ra_c) * K + sk);
  {
    const int rw_c = sn < N - 1 - n0 ? sn : N - 1 - n0;
    stage_load(pb, W + (size_t)(n0 + rw_c) * K + sk);
  }

  int buf = 0;
  for (int c0 = 0; c0 < K; c0 += BK) {
    stage_store(&s_a[buf][sn * ASTR + sk], pa);
    stage_store(&s_b[buf][sn * WSTR + sk], pb);
    __syncthreads();

    // One K tile ahead in registers: the loads for c0+BK retire while this tile's WMMAs run, and
    // are published on the next iteration. One __syncthreads per tile, because the store lands in
    // the other buffer.
    if (c0 + BK < K) {
      stage_load(pa, A + (size_t)(m0 + ra_c) * K + c0 + BK + sk);
      const int rw_c = sn < N - 1 - n0 ? sn : N - 1 - n0;
      stage_load(pb, W + (size_t)(n0 + rw_c) * K + c0 + BK + sk);
    }

#pragma unroll
    for (int ks = 0; ks < BK; ks += 16) {
      const i2v af = frag(s_a[buf], ASTR, wrow, ks, lane);
      i2v bf[NT];
#pragma unroll
      for (int j = 0; j < NT; ++j) bf[j] = frag(s_b[buf], WSTR, j * 16, ks, lane);
#pragma unroll
      for (int j = 0; j < NT; ++j)
        raw[j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af, bf[j], raw[j]);

      // End of a 128-column scale block: fold both scales in and reset. `ks` is compile-time after
      // unrolling but `c0` is not, so this stays a runtime test on purpose.
      if ((c0 + ks + 16) % KBLK == 0 || c0 + ks + 16 >= K) {
        const int kb = (c0 + ks) / KBLK;
        // FN = BN = 128, so the weight's block scale is one scalar for the whole block.
        const float ws = Bs[(size_t)nb * kgrp + kb];
        float as[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) {
          // Accumulator slot t holds row (lane >> 4) * 8 + t of this wave's 16-row tile.
          const int r = m0 + wrow + ((lane >> 4) * 8 + t);
          as[t] = r < M ? ws * As[(size_t)r * kgrp + kb] : 0.f;
        }
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
          for (int t = 0; t < 8; ++t) { out[j][t] += raw[j][t] * as[t]; raw[j][t] = 0.f; }
      }
    }
    buf ^= 1;
  }

#pragma unroll
  for (int j = 0; j < NT; ++j)
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      const int m = m0 + wrow + ((lane >> 4) * 8 + t);
      const int n = n0 + j * 16 + (lane & 15);
      if (m < M && n < N) C[(size_t)m * N + n] = (__bf16)out[j][t];
    }
}

static void launch(uintptr_t a, uintptr_t w, uintptr_t asc, uintptr_t bsc, uintptr_t c,
                   int M, int N, int K, uintptr_t stream) {
  if (M <= 0) throw std::runtime_error("radiance_fp8_block_wmma: M must be positive");
  if (K % KBLK) throw std::runtime_error("radiance_fp8_block_wmma: K must be a multiple of 128");
  if (N % BN) throw std::runtime_error("radiance_fp8_block_wmma: N must be a multiple of 128");
  const unsigned char *A = (const unsigned char *)a;
  const unsigned char *W = (const unsigned char *)w;
  const float *As = (const float *)asc;
  const float *Bs = (const float *)bsc;
  __bf16 *C = (__bf16 *)c;
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  hipLaunchKernelGGL((radiance_fp8_block_wmma_gemm), grid, dim3(NTHREADS), 0,
                     (hipStream_t)stream, A, W, As, Bs, C, M, N, K);
}

PYBIND11_MODULE(radiance_fp8_block_wmma, m) {
  m.def("launch", &launch, "block-FP8 (W8A8, 128x128) fp8-WMMA GEMM (gfx1201)");
}
