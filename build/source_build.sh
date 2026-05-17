#!/usr/bin/env bash
# Build PyTorch 2.10.0 from source with TORCH_CUDA_ARCH_LIST="12.1;12.1a" —
# produces the first wheel on this DGX Spark hardware with native sm_121 and
# sm_121a (arch-specific feature) cubins.
# Artifacts isolated in docker named volume dgx-spark-build-strict.
set -euo pipefail

WORK_VOL=dgx-spark-build-strict
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker volume create "$WORK_VOL" >/dev/null

docker run --rm --gpus all --ipc=host --shm-size=8g \
  -v "$WORK_VOL":/work \
  -v "$REPO":/repo:ro \
  -w /work \
  nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04 \
  bash -c '
    set -euxo pipefail
    export DEBIAN_FRONTEND=noninteractive

    apt-get update -qq
    apt-get install -y -qq \
      build-essential cmake ninja-build git ca-certificates \
      python3 python3-dev python3-venv python3-pip \
      libopenblas-dev libomp-dev libnuma-dev libssl-dev zlib1g \
      ccache mold \
      cudnn9-cuda-13-2 \
      cusparselt-cuda-13 libcusparselt0-cuda-13 libcusparselt0-dev-cuda-13

    if [[ ! -d /work/pytorch/.git ]]; then
      git clone --recursive --branch v2.10.0 \
        https://github.com/pytorch/pytorch.git /work/pytorch
    fi
    cd /work/pytorch
    git submodule update --init --recursive

    git checkout -- .
    git -C third_party/flash-attention checkout -- .
    # Drop stale CMakeCache (LDFLAGS_INIT and other env-derived vars are cached
    # from the first configure; env-var changes are ignored on re-runs).
    rm -rf /work/pytorch/build

    export USE_CUDA=1 USE_CUDNN=1 USE_CUBLAS=1 USE_CUSPARSELT=1
    export USE_CUFILE=0 USE_MPI=0 USE_DISTRIBUTED=0 USE_NCCL=0
    export USE_TENSORPIPE=0 USE_GLOO=0 USE_NNPACK=0
    export USE_FBGEMM=0 USE_FBGEMM_GENAI=0 USE_MAGMA=0
    export USE_KINETO=0 USE_MKLDNN=0 USE_ITT=0
    export USE_FLASH_ATTENTION=1 USE_MEM_EFF_ATTENTION=1
    export BUILD_TEST=0 BUILD_CAFFE2=0
    export TORCH_CUDA_ARCH_LIST="12.1;12.1a"
    export USE_SYSTEM_LIBS=0 BLAS=OpenBLAS
    export CMAKE_GENERATOR=Ninja
    export MAX_JOBS=$(nproc)
    export PYTORCH_BUILD_VERSION=2.10.0 PYTORCH_BUILD_NUMBER=1
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    mkdir -p /work/.ccache
    export CCACHE_DIR=/work/.ccache CCACHE_MAXSIZE=20G
    export CC=gcc CXX=g++
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
    export CMAKE_ASM_COMPILER_LAUNCHER=ccache

    python3 -m venv /tmp/v
    . /tmp/v/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt -r requirements-build.txt
    pip wheel . -w dist --no-deps --verbose
    pip install dist/torch-*.whl

    # cd away from /work/pytorch so Python imports the installed wheel, not the
    # source tree (the torch/ subdir would shadow the installed package).
    cd /tmp
    python -c "import torch; print(\"torch.__version__:\", torch.__version__); print(\"torch.cuda.get_arch_list():\", torch.cuda.get_arch_list())"
  '

echo "Wheel built in volume $WORK_VOL. Run bench with: bash $REPO/bench/run_bakeoff.sh"
