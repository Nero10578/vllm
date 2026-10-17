// RADIANCE gfx1201 dense block-FP8 GEMM: C[M,N] bf16 = (A * As) . (W * Bs)^T
//
//   A   [M,K]        e4m3, row-major
//   W   [N,K]        e4m3, row-major (the compressed-tensors / FP8 block weight)
//   As  [M, K/128]   fp32, activation scales, group shape (1,128)
//   Bs  [N/128,K/128] fp32, weight scales,  block shape (128,128)
//   C   [M,N]        bf16
//
// Why this exists: vLLM's ROCm block-FP8 route is the Triton kernel
// (_w8a8_triton_block_scaled_mm, `accumulator += tl.dot(a, b) * a_s * b_s`). On gfx1201 Triton will
// not emit `v_wmma_f32_16x16x16_fp8_fp8`; it upconverts and runs the 16-bit WMMA. The tuned Triton
// config is excellent at that -- measured here (radiance/gate_fp8_block.py) at 139-159 TFLOP/s,
// i.e. ~87-99% of the 160.2 bf16 WMMA ceiling, so it has no headroom left, while the fp8
// instruction runs at 325.2. This kernel is the same math on the fp8 instruction; the ceiling it is
// chasing is the fp8 one.
//
// Numerics: this is a pure re-implementation of the Triton kernel's semantics -- the same fp32
// accumulation, the same per-(1,128) activation scale and per-(128,128) weight scale applied inside
// the K loop. Output is not required to be bit-identical (the accumulation order differs), so it is
// gated by a relative-error check against the Triton path and by RADIANCE_FP8_WMMA.
//
// Layout reused verbatim from the proven MXFP4 fp8-WMMA kernel (radiance_mxfp4_fp8.hip), whose
// fragment mapping was confirmed empirically there (0/256 elements wrong on a 16x16x16 tile):
//   A: lane l holds A[l%16][(l/16)*8+j] | B: B[(l/16)*8+j][l%16] | C: C[(l/16)*8+j][l%16]
// The two differences are (a) W arrives already in e4m3, so the staging load is a plain copy
// instead of the e2m1->e4m3 table fold, and (b) the per-block rescale is by the ACCUMULATOR ROW's
// activation scale (a_s is per m), where the MXFP4 kernel's scale was per column (per n).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
#include <cstdlib>
namespace py = pybind11;

typedef float floatx8 __attribute__((ext_vector_type(8)));
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));

#define BN 64
#define BK 64
#define PAD 8
#define ASTR (BK + PAD)
#define WSTR (BK + PAD)
#define WG_K 128          // weight k-block (and activation group along k)
#define WG_N 128          // weight n-block

// BMF = rows per CTA. Two instantiations: 128 for the prefill band, 64 for M in [64,128). The wave
// grid is 2 N-waves x (BMF/32) M-waves, so the accumulator is BMF*BN/(NWAVE*32) = 32 floats/lane
// either way and the tile never issues WMMAs for more rows than the block is tall.
template <int BMF>
__global__ __launch_bounds__((BMF / 32) * 2 * 32) void radiance_fp8_block_wmma_gemm(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ W,
    const float *__restrict__ As, const float *__restrict__ Bs,
    __bf16 *__restrict__ C, int M, int N, int K) {
  constexpr int NWAVE = (BMF / 32) * 2;
  constexpr int NTHREADS = NWAVE * 32;
  __shared__ unsigned char sA[BMF * ASTR];
  __shared__ unsigned char sW[BN * WSTR];
  __shared__ float sAs[BMF];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave >> 1, wn = wave & 1;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;

  // Grouped-M CTA swizzle (Triton's GROUP_SIZE_M). Walking N at fixed M -- the launch order --
  // means every M-block re-streams the whole [N,K] weight from DRAM, because the weight does not
  // fit the 8 MB L2: measured at M=2496 that is (M/BMF) * N * K bytes, e.g. 20 * 57.8 MB = 1156 MB
  // for gate_up, 1820 us at 635 GB/s, which is exactly where both this kernel and Triton sat.
  // Eight M-blocks grouped per weight slab keep the group's A tiles (8*128*K = 5.5 MB) plus one
  // BN*K slab (344 KB) inside L2, so each weight byte is fetched once per group instead of once
  // per M-block. A bijection on the tile grid, so it cannot change the arithmetic.
  const int num_pid_m = (M + BMF - 1) / BMF;
  const int num_pid_n = (N + BN - 1) / BN;
  constexpr int GROUP_M = 8;
  const int pid = blockIdx.x + blockIdx.y * gridDim.x;
  const int in_group = pid % (GROUP_M * num_pid_n);
  const int first_m = (pid / (GROUP_M * num_pid_n)) * GROUP_M;
  int gsz = num_pid_m - first_m;
  if (gsz > GROUP_M) gsz = GROUP_M;
  const int m0 = (first_m + in_group % gsz) * BMF, n0 = (in_group / gsz) * BN;

  const int kgrp = K / WG_K;          // K/128 scale groups
  const int nb = n0 / WG_N;           // this CTA's weight n-block (BN=64 < WG_N=128, so it is unique)

  floatx8 acc[2][2], tmp[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) { acc[i][j][e] = 0.f; tmp[i][j][e] = 0.f; }

  // Register-staged double buffering of the K loop. Previously a slab's A and W went straight from
  // global into LDS and then __syncthreads(), so the entire L2/DRAM latency of those loads was
  // exposed once per slab -- and a slab is only four WMMA steps, on two CTAs per CU. Now the next
  // slab is fetched into registers while the current slab computes and published to LDS only after
  // the trailing barrier. Costs 2 uint4 for A and 1-2 for W.
  constexpr int WPASSES = (BN * BK >= NTHREADS * 32) ? 2 : 1;
  uint4_t Au0, Au1, Wu0, Wu1;
  // Clamp, never predicate: a predicated staging load gets its own exec-mask region and defeats the
  // counted load scheduling (see the MXFP4 kernel's note). Rows past M/N are dropped in the epilogue.
#define LOAD_SLAB(kk0)                                                                       \
  do {                                                                                       \
    const int a0 = tid * 16, ar = a0 / BK, ac = a0 % BK;                                     \
    const int arc = ar < M - 1 - m0 ? ar : M - 1 - m0;                                       \
    Au0 = *(const uint4_t *)(A + (size_t)(m0 + arc) * K + (kk0) + ac);                        \
    const int a1 = (NTHREADS + tid) * 16, br = a1 / BK, bc = a1 % BK;                         \
    const int brc = br < M - 1 - m0 ? br : M - 1 - m0;                                       \
    Au1 = *(const uint4_t *)(A + (size_t)(m0 + brc) * K + (kk0) + bc);                        \
    const int w0 = tid * 16, wr = w0 / BK, wc = w0 % BK;                                     \
    const int wrc = wr < N - 1 - n0 ? wr : N - 1 - n0;                                       \
    Wu0 = *(const uint4_t *)(W + (size_t)(n0 + wrc) * K + (kk0) + wc);                        \
    if (WPASSES > 1) {                                                                       \
      const int w1 = (NTHREADS + tid) * 16, xr = w1 / BK, xc = w1 % BK;                       \
      const int xrc = xr < N - 1 - n0 ? xr : N - 1 - n0;                                     \
      Wu1 = *(const uint4_t *)(W + (size_t)(n0 + xrc) * K + (kk0) + xc);                      \
    }                                                                                        \
  } while (0)

  LOAD_SLAB(0);

  for (int k0 = 0; k0 < K; k0 += BK) {
    {
      const int a0 = tid * 16, r0 = a0 / BK, c0 = a0 % BK;
      *(uint4_t *)(&sA[r0 * ASTR + c0]) = Au0;
      const int a1 = (NTHREADS + tid) * 16, r1 = a1 / BK, c1 = a1 % BK;
      *(uint4_t *)(&sA[r1 * ASTR + c1]) = Au1;
    }
    {
      const int w0 = tid * 16, r0 = w0 / BK, c0 = w0 % BK;
      *(uint4_t *)(&sW[r0 * WSTR + c0]) = Wu0;
      if (WPASSES > 1) {
        const int w1 = (NTHREADS + tid) * 16, r1 = w1 / BK, c1 = w1 % BK;
        *(uint4_t *)(&sW[r1 * WSTR + c1]) = Wu1;
      }
    }
    // Activation scales for this CTA's rows at the k-group k0/128 (two BK slabs share a group).
    const int kb = k0 / WG_K;
    if (tid < BMF) {
      const int rc = tid < M - 1 - m0 ? tid : M - 1 - m0;
      sAs[tid] = As[(size_t)(m0 + rc) * kgrp + kb];
    }
    __syncthreads();

    // Issue the next slab's global loads now, before the matrix work: they retire into registers
    // while this slab computes and are published to LDS on the next iteration.
    if (k0 + BK < K) LOAD_SLAB(k0 + BK);

    // One weight scale for the whole CTA: BN=64 columns lie inside a single 128-wide n-block.
    const float bsc = Bs[(size_t)nb * kgrp + kb];

    // One-step software pipeline. Each 16-wide step issues only four WMMAs, so a load-then-barrier
    // -then-WMMA sequence exposes the whole LDS latency every step and the matrix pipe idles; the
    // MXFP4 kernel hid the same latency by amortising it over eight WMMAs per step. Prefetch step
    // s+1's fragments while step s issues, and let the scheduler interleave (no blanket
    // sched_barrier -- the explicit two-stage structure already bounds hoisting to one step).
    int2_t af[2], wf[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const unsigned char *p = &sA[(wm * 32 + i * 16 + col) * ASTR + kb8];
      af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const unsigned char *p = &sW[(wn * 32 + j * 16 + col) * WSTR + kb8];
      wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
    }

#pragma unroll
    for (int step = 0; step < BK / 16; ++step) {
      int2_t naf[2], nwf[2];
      if (step + 1 < BK / 16) {
        const int nk = (step + 1) * 16 + kb8;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const unsigned char *p = &sA[(wm * 32 + i * 16 + col) * ASTR + nk];
          naf[i][0] = *(const int *)p; naf[i][1] = *(const int *)(p + 4);
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const unsigned char *p = &sW[(wn * 32 + j * 16 + col) * WSTR + nk];
          nwf[j][0] = *(const int *)p; nwf[j][1] = *(const int *)(p + 4);
        }
      }
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          tmp[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], tmp[i][j]);
      if (step + 1 < BK / 16) {
#pragma unroll
        for (int i = 0; i < 2; ++i) { af[i][0] = naf[i][0]; af[i][1] = naf[i][1]; }
#pragma unroll
        for (int j = 0; j < 2; ++j) { wf[j][0] = nwf[j][0]; wf[j][1] = nwf[j][1]; }
      }
    }
    // The block scale is constant over the whole WG_K=128 group, which spans TWO BK=64 slabs. So
    // accumulate both slabs into tmp and fold them into acc once, on the group's second slab:
    // (p0 + p1) * s == p0*s + p1*s. Doing this per 16-wide step (the first version) cost an 8x
    // rescale + LDS read per WMMA and made the kernel epilogue-bound -- measured 0.55-0.82x of
    // Triton at every large-M shape even though the matrix math was correct.
    if ((k0 % WG_K) == BK) {
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            acc[i][j][e] += tmp[i][j][e] * sAs[wm * 32 + i * 16 + kb8 + e] * bsc;
            tmp[i][j][e] = 0.f;
          }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * 32 + i * 16 + kb8 + e;
        const int n = n0 + wn * 32 + j * 16 + col;
        if (m < M && n < N) C[(size_t)m * N + n] = (__bf16)acc[i][j][e];
      }
}

static void launch(uintptr_t a, uintptr_t w, uintptr_t asc, uintptr_t bsc, uintptr_t c,
                   int M, int N, int K, uintptr_t stream) {
  if (M <= 0) throw std::runtime_error("radiance_fp8_block_wmma: M must be positive");
  if (K % WG_K) throw std::runtime_error("radiance_fp8_block_wmma: K must be a multiple of 128");
  if (N % 16) throw std::runtime_error("radiance_fp8_block_wmma: N must be a multiple of 16");
  const unsigned char *A = (const unsigned char *)a;
  const unsigned char *W = (const unsigned char *)w;
  const float *As = (const float *)asc;
  const float *Bs = (const float *)bsc;
  __bf16 *C = (__bf16 *)c;
  if (M >= 128) {
    dim3 grid((N + BN - 1) / BN, (M + 127) / 128);
    hipLaunchKernelGGL((radiance_fp8_block_wmma_gemm<128>), grid, dim3(8 * 32), 0,
                       (hipStream_t)stream, A, W, As, Bs, C, M, N, K);
  } else {
    dim3 grid((N + BN - 1) / BN, (M + 63) / 64);
    hipLaunchKernelGGL((radiance_fp8_block_wmma_gemm<64>), grid, dim3(4 * 32), 0,
                       (hipStream_t)stream, A, W, As, Bs, C, M, N, K);
  }
}

PYBIND11_MODULE(radiance_fp8_block_wmma, m) {
  m.def("launch", &launch, "block-FP8 (W8A8, 128x128) fp8-WMMA GEMM (gfx1201)");
}
