# Build an nvbench-linked GEMM benchmark for sm_121 cross-validation
# of the Python harness. nvbench provides built-in CUDA-event timing,
# L2 flush, and statistical sampling — if our Python numbers diverge
# from nvbench's by >3%, we have a bug in _harness.py.
cmake_minimum_required(VERSION 3.20)
project(sm121_gemm LANGUAGES CXX CUDA)

# GB10 only. No PTX fallback — failure here is an explicit signal that nvbench
# doesn't yet support sm_121 (per Risk Register), not a silently degraded build.
set(CMAKE_CUDA_ARCHITECTURES 121)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Pull nvbench. Pinned to a recent main commit (2026-05) since v0.2.0 (Feb 2026)
# is the last tagged release and we want fixes for Blackwell SASS metrics.
include(FetchContent)
FetchContent_Declare(
  nvbench
  GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
  GIT_TAG        main
  GIT_SHALLOW    TRUE
)
set(NVBench_ENABLE_EXAMPLES OFF CACHE BOOL "")
set(NVBench_ENABLE_TESTING  OFF CACHE BOOL "")
FetchContent_MakeAvailable(nvbench)

find_package(CUDAToolkit REQUIRED COMPONENTS cublas cudart)

add_executable(sm121_gemm sm121_gemm.cu)
target_link_libraries(sm121_gemm
  PRIVATE
    nvbench::main
    CUDA::cublas
    CUDA::cudart
)
target_compile_options(sm121_gemm PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
