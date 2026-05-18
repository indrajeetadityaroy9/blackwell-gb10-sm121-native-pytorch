#!/usr/bin/env bash
# Build the sm_121 nvbench GEMM binary inside a CUDA-devel container.
# Output: /work/nvbench_shim/build/sm121_gemm (dgx-spark-build-strict volume).
#
# Invoked by bench/nvbench_shim/run_nvbench.py via docker run.

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq >/dev/null
# Ubuntu 24.04 ships cmake 3.28; nvbench main needs 3.30+ — install via pip.
apt-get install -y -qq ninja-build git ca-certificates python3 python3-pip python3-venv >/dev/null
python3 -m venv /tmp/cmakeenv
. /tmp/cmakeenv/bin/activate
pip install -q --upgrade pip
pip install -q 'cmake>=3.31'
which cmake; cmake --version | head -1

BUILD_DIR=/work/nvbench_shim/build
mkdir -p "$BUILD_DIR"

# FetchContent clones nvbench into build/_deps/nvbench-src
echo "[build] configuring with CMAKE_CUDA_ARCHITECTURES=121"
cmake -S /repo/bench/nvbench_shim -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  2>&1 | tail -30

# Build
echo "[build] compiling sm121_gemm"
cmake --build "$BUILD_DIR" --target sm121_gemm 2>&1 | tail -20

BIN="$BUILD_DIR/sm121_gemm"
if [[ ! -x "$BIN" ]]; then
  echo "[build] FATAL: build did not produce $BIN" >&2
  exit 2
fi
echo "[build] OK: $BIN"
ls -lh "$BIN"
