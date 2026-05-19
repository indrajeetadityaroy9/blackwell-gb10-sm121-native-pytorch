#!/usr/bin/env bash
# Wipe every Docker volume the bake-off creates. Prompts for confirmation
# unless -y/--force is passed (CI / automation).
#
# Volumes:
#   dgx-spark-build-strict        wheel build artifacts (~25 GB)
#   dgx-spark-hf-cache            HuggingFace model cache (~80-170 GB)
#   dgx-spark-uv-cache            uv package cache (~5 GB)
#   dgx-spark-apt-cache           apt cache for nsight-compute (~2 GB)
#   dgx-spark-triton-cache-{a,b,c}  Triton JIT cache, per-wheel (~500 MB each)
#   dgx-spark-cutlass-jit-{a,b,c}   flash_attn.cute CuTe-DSL JIT (~1 GB each)

set -euo pipefail

VOLUMES=(
  dgx-spark-build-strict
  dgx-spark-hf-cache
  dgx-spark-uv-cache
  dgx-spark-apt-cache
  dgx-spark-triton-cache-a dgx-spark-triton-cache-b dgx-spark-triton-cache-c
  dgx-spark-cutlass-jit-a dgx-spark-cutlass-jit-b dgx-spark-cutlass-jit-c
)

echo "About to remove these Docker volumes:"
printf '  %s\n' "${VOLUMES[@]}"

if [[ "${1:-}" == "-y" || "${1:-}" == "--force" ]]; then
  ans="y"
else
  read -r -p "Confirm [y/N]: " ans
fi
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "Aborted."; exit 0; }

for v in "${VOLUMES[@]}"; do
  if docker volume rm "$v" 2>/dev/null; then
    echo "  removed $v"
  else
    echo "  (skipped $v — not present)"
  fi
done
