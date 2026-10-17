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

  for (int k0 = 0; k0 < K; k0 += BK) {
    {   // A tile: BMF*BK bytes, 32 B per thread. Clamp, never predicate -- a predicated staging
        // load gets its own exec-mask region and defeats the counted s_wait_loadcnt (see the
        // MXFP4 kernel's note); the epilogue already drops rows past M.
      const int r = tid >> 1, off = (tid & 1) * 32;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;
      const unsigned char *src = A + (size_t)(m0 + rc) * K + k0 + off;
      *(uint4_t *)(&sA[r * ASTR + off]) = *(const uint4_t *)(src);
      *(uint4_t *)(&sA[r * ASTR + off + 16]) = *(const uint4_t *)(src + 16);
    }
    {   // W tile: BN*BK bytes, 16 B per thread per pass. Already e4m3 -- plain copy, no fold.
        // Loop, not a fixed `r = tid >> 2`: the tile is 4096 B and the BMF=64 instantiation has
        // only 128 threads, so a single-pass form would stage half of it.
      for (int idx = tid * 16; idx < BN * BK; idx += NTHREADS * 16) {
        const int r = idx / BK, c = idx % BK;
        const int rc = r < N - 1 - n0 ? r : N - 1 - n0;
        *(uint4_t *)(&sW[r * WSTR + c]) =
            *(const uint4_t *)(W + (size_t)(n0 + rc) * K + k0 + c);
      }
    }
    // Activation scales for this CTA's rows at the k-group k0/128 (two BK slabs share a group).
    const int kb = k0 / WG_K;
    if (tid < BMF) {
      const int rc = tid < M - 1 - m0 ? tid : M - 1 - m0;
      sAs[tid] = As[(size_t)(m0 + rc) * kgrp + kb];
    }
    __syncthreads();

    // One weight scale for the whole CTA: BN=64 columns lie inside a single 128-wide n-block.
    const float bsc = Bs[(size_t)nb * kgrp + kb];

#pragma unroll
    for (int step = 0; step < BK / 16; ++step) {
      const int kk = step * 16 + kb8;
      int2_t af[2], wf[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const unsigned char *p = &sA[(wm * 32 + i * 16 + col) * ASTR + kk];
        af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const unsigned char *p = &sW[(wn * 32 + j * 16 + col) * WSTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      // Scheduling barrier per k-step, or the compiler hoists every step's fragment loads above
      // the first WMMA and holds four steps of fragments live (see the MXFP4 kernel).
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
          tmp[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], tmp[i][j]);
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
