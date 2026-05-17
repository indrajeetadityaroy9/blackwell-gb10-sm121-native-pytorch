#!/usr/bin/env bash
# 3-way performance comparison for DGX Spark (Grace + Blackwell GB10).
#
#  A : PyPI baseline torch==2.9.0+cu130    (nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04)
#  B : Source-built wheel with native sm_121 cubins   (from docker volume dgx-spark-build-strict)
#  C : NGC vendor reference                (nvcr.io/nvidia/pytorch:26.04-py3)
#
# Run B is SKIPPED if the volume has no wheel — produce it first with build/source_build.sh.
#
# Output: bench/logs/{02_runA.log, 03_runB.log, 04_runC.log, SUMMARY.txt}

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$REPO_DIR/bench/logs"
mkdir -p "$LOGS"

ts()  { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*"; }

# Run A : PyPI baseline
log " Run A : PyPI torch==2.9.0+cu130 + triton"
docker run --rm --gpus all --ipc=host --shm-size=4g \
  -v "$REPO_DIR":/repo:ro \
  nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04 \
  bash -c '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq python3 python3-venv python3-pip ca-certificates >/dev/null
    python3 -m venv /tmp/v && . /tmp/v/bin/activate
    pip install --upgrade pip 2>&1 | tail -1
    pip install --extra-index-url https://download.pytorch.org/whl/cu130 torch==2.9.0+cu130 2>&1 | tail -1
    pip install triton 2>&1 | tail -1 || true
    python /repo/bench/bench_full.py
  ' >"$LOGS/02_runA.log" 2>&1
A_RC=$?
log "Run A exit: ${A_RC}"

# Run B : source-built wheel from docker volume
if docker run --rm -v dgx-spark-build-strict:/v alpine \
     sh -c 'ls /v/pytorch/dist/torch-*.whl 2>/dev/null | head -1' \
   | grep -q '.whl'; then
  log " Run B : source-built wheel (dgx-spark-build-strict) "
  docker run --rm --gpus all --ipc=host --shm-size=4g \
    -v dgx-spark-build-strict:/work:ro \
    -v "$REPO_DIR":/repo:ro \
    nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04 \
    bash -c '
      set -euo pipefail
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq
      apt-get install -y -qq python3 python3-venv python3-pip ca-certificates \
        libopenblas0 libnuma1 cudnn9-cuda-13-2 \
        cusparselt-cuda-13 libcusparselt0-cuda-13 >/dev/null
      python3 -m venv /tmp/v && . /tmp/v/bin/activate
      pip install --upgrade pip 2>&1 | tail -1
      WHEEL=$(ls /work/pytorch/dist/torch-*.whl | head -1)
      pip install "$WHEEL" 2>&1 | tail -1
      python -c "import torch; print(\"arch_list:\", torch.cuda.get_arch_list())"
      python /repo/bench/bench_full.py
    ' >"$LOGS/03_runB.log" 2>&1
  B_RC=$?
  log "Run B exit: ${B_RC}"
else
  log "Run B : SKIPPED (no wheel in volume dgx-spark-build-strict — build first with build/source_build.sh) ==="
  B_RC=skipped
fi

#  Run C : NGC vendor reference
log "Run C : NGC nvcr.io/nvidia/pytorch:26.04-py3 "
docker run --rm --gpus all --ipc=host --shm-size=4g \
  -v "$REPO_DIR":/repo:ro \
  nvcr.io/nvidia/pytorch:26.04-py3 \
  python /repo/bench/bench_full.py >"$LOGS/04_runC.log" 2>&1
C_RC=$?
log "Run C exit: ${C_RC}"

# Summary
{
  echo "DGX Spark 3-way comparison"
  echo "Host: $(hostname)   Date: $(ts)"
  echo
  for entry in "02_runA:A — PyPI torch 2.9.0+cu130 + triton" \
               "03_runB:B — Source-built wheel (native sm_121)" \
               "04_runC:C — NGC pytorch:26.04-py3 (vendor reference)"; do
    fn="${entry%%:*}"; label="${entry#*:}"
    f="$LOGS/${fn}.log"
    [[ -f "$f" ]] || continue
    echo "--- ${label} ---"
    awk "/^Summary {flag=1; next} flag" "$f" | head -20
    if ! grep -q "^ Summary" "$f"; then
      echo "(no Summary section — last 15 lines:)"
      tail -15 "$f" | sed 's/^/    /'
    fi
    echo
  done
  echo "Return codes: A=${A_RC}  B=${B_RC}  C=${C_RC}"
} | tee "$LOGS/SUMMARY.txt"

chown -R indrajeetadityaroy:indrajeetadityaroy "$LOGS" 2>/dev/null || true
log "Done. See $LOGS/SUMMARY.txt"
