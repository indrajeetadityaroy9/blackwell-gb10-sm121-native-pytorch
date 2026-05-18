#!/usr/bin/env bash
# Source-build FlashInfer for sm_121 inside a CUDA-devel container.
# PyPI wheels are sm_120-only; trtllm-gen FMHA cubins lack sm_121 (TRT-LLM #11799).
# Running sm_100 cubins on sm_121 raises cudaErrorIllegalInstruction.
#
# Build output: a wheel in /work/flashinfer/dist/ inside the dgx-spark-build-strict
# volume, alongside the PyTorch wheel. Re-mount that volume read-only when running
# Tier 4 tests.
#
# Invocation:
#   sg docker -c 'docker run --rm --gpus all \
#     -v dgx-spark-build-strict:/work \
#     -v $PWD:/repo:ro \
#     nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04 \
#     bash /repo/bench/experimental/build_flashinfer.sh'
#
# Time budget: ~30-60 min cold build, ~5-10 min incremental via ccache.

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq >/dev/null
apt-get install -y -qq \
  python3 python3-venv python3-pip ca-certificates git \
  build-essential cmake ninja-build ccache \
  libopenblas0 libnuma1 cudnn9-cuda-13-2 \
  cusparselt-cuda-13 libcusparselt0-cuda-13 >/dev/null

mkdir -p /work/flashinfer /work/.ccache_flashinfer
export CCACHE_DIR=/work/.ccache_flashinfer
export CCACHE_MAXSIZE=10G

# Pull / refresh source — pinned to flashinfer v0.5.2+ which first targets sm_12x
if [[ ! -d /work/flashinfer/.git ]]; then
  echo "[build] cloning flashinfer"
  git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /work/flashinfer
fi
cd /work/flashinfer
git checkout v0.5.2 2>/dev/null || git checkout main
git submodule update --init --recursive

python3 -m venv /tmp/v && . /tmp/v/bin/activate
pip install -q --upgrade pip

# Install the PyTorch wheel we built earlier so flashinfer can link against it
WHEEL=$(ls /work/pytorch/dist/torch-*.whl 2>/dev/null | head -1)
if [[ -z "$WHEEL" ]]; then
  echo "[build] FATAL: no PyTorch wheel in /work/pytorch/dist — build PyTorch first" >&2
  exit 2
fi
pip install -q "$WHEEL"
pip install -q ninja packaging wheel setuptools

# Build wheel with native sm_121
export TORCH_CUDA_ARCH_LIST="12.1"
export MAX_JOBS=$(nproc)
export CC=gcc CXX=g++
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache

echo "[build] building flashinfer wheel for sm_121 (this takes ~30-60 min cold)"
pip wheel . -w dist --no-deps --no-build-isolation --verbose 2>&1 | tail -50

WHL=$(ls dist/flashinfer*.whl | head -1)
if [[ -z "$WHL" ]]; then
  echo "[build] FATAL: no wheel produced in dist/" >&2
  exit 2
fi
echo "[build] OK: $WHL"
ls -lh "$WHL"
