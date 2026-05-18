// nvbench shim: FP16 GEMM 8192^3 via cuBLAS, used to cross-validate the
// Python harness in bench/_harness.py. If the two report numbers differing
// by more than 3%, the Python harness has a bug — either in CUDA event
// ordering, L2 flush logic, or FLOPs accounting.
//
// We deliberately do NOT use nvbench's built-in clock-lock feature
// (`set_blocking_kernel_timeout` etc.) because the bake-off already holds
// the lock via the dgx-bench-clocklock container. Double-locking would
// conflict and is harmless but unnecessary.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <nvbench/nvbench.cuh>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t _e = (x);                                                      \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                   cudaGetErrorString(_e), __FILE__, __LINE__);                \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

#define CKB(x)                                                                 \
  do {                                                                         \
    cublasStatus_t _s = (x);                                                   \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                         \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)_s, __FILE__,    \
                   __LINE__);                                                  \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

static void fp16_gemm(nvbench::state &state) {
  const int M = static_cast<int>(state.get_int64("M"));
  const int N = static_cast<int>(state.get_int64("N"));
  const int K = static_cast<int>(state.get_int64("K"));

  // Allocate row-major fp16 A (M,K), B (K,N), C (M,N).
  // cuBLAS is column-major-native, but cublasLt with order=ROW handles it.
  __half *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CK(cudaMalloc(&dA, sizeof(__half) * M * K));
  CK(cudaMalloc(&dB, sizeof(__half) * K * N));
  CK(cudaMalloc(&dC, sizeof(__half) * M * N));
  CK(cudaMemset(dA, 0x3c, sizeof(__half) * M * K)); // ~1.0 in fp16
  CK(cudaMemset(dB, 0x3c, sizeof(__half) * K * N));

  cublasLtHandle_t lt = nullptr;
  CKB(cublasLtCreate(&lt));

  // Operation: C = alpha * A @ B + beta * C with fp16 in/out, fp32 accumulate.
  cublasLtMatmulDesc_t opDesc = nullptr;
  CKB(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

  cublasLtMatrixLayout_t aL = nullptr, bL = nullptr, cL = nullptr;
  CKB(cublasLtMatrixLayoutCreate(&aL, CUDA_R_16F, M, K, M));
  CKB(cublasLtMatrixLayoutCreate(&bL, CUDA_R_16F, K, N, K));
  CKB(cublasLtMatrixLayoutCreate(&cL, CUDA_R_16F, M, N, M));

  cublasLtMatmulPreference_t pref = nullptr;
  CKB(cublasLtMatmulPreferenceCreate(&pref));
  const size_t ws_bytes = 1ULL << 28; // 256 MB workspace
  void *ws = nullptr;
  CK(cudaMalloc(&ws, ws_bytes));
  CKB(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &ws_bytes, sizeof(ws_bytes)));

  cublasLtMatmulHeuristicResult_t heur = {};
  int returned = 0;
  CKB(cublasLtMatmulAlgoGetHeuristic(lt, opDesc, aL, bL, cL, cL, pref, 1,
                                     &heur, &returned));
  if (returned == 0) {
    std::fprintf(stderr, "no cuBLASLt algo for fp16 GEMM %dx%dx%d on sm_121\n",
                 M, N, K);
    std::exit(2);
  }

  const float alpha = 1.0f, beta = 0.0f;

  // FLOPs accounting for nvbench (so it can report TFLOP/s).
  state.add_element_count(static_cast<int64_t>(M) * N * K, "MNK");
  state.add_global_memory_reads<__half>(static_cast<int64_t>(M) * K +
                                        static_cast<int64_t>(K) * N);
  state.add_global_memory_writes<__half>(static_cast<int64_t>(M) * N);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch &launch) {
    CKB(cublasLtMatmul(lt, opDesc, &alpha, dA, aL, dB, bL, &beta, dC, cL, dC,
                       cL, &heur.algo, ws, ws_bytes, launch.get_stream()));
  });

  CK(cudaFree(ws));
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(aL);
  cublasLtMatrixLayoutDestroy(bL);
  cublasLtMatrixLayoutDestroy(cL);
  cublasLtMatmulDescDestroy(opDesc);
  cublasLtDestroy(lt);
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
}

NVBENCH_BENCH(fp16_gemm)
    .add_int64_axis("M", {8192})
    .add_int64_axis("N", {8192})
    .add_int64_axis("K", {8192});
