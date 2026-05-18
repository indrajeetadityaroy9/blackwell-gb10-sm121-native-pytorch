# nvbench-linked GEMM benchmark for cross-validating the Python harness.
# Divergence > 3% from nvbench means a bug in _harness.py.
cmake_minimum_required(VERSION 3.20)
project(sm121_gemm LANGUAGES CXX CUDA)

# GB10 only — no PTX fallback. A build failure flags missing sm_121 support.
set(CMAKE_CUDA_ARCHITECTURES 121)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# nvbench main (post-v0.2.0) for Blackwell SASS metric fixes.
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
