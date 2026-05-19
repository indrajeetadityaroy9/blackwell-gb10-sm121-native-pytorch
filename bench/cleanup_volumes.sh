#!/usr/bin/env bash
# Wipe every Docker volume the bake-off creates. Prompts for confirmation
# unless -y/--force is passed (CI / automation).
#
# Volumes:
#   dgx-spark-build-strict          source-built torch wheel artifact (~25 GB)
#   dgx-spark-hf-cache              HuggingFace model cache (~16 GB Llama-3-8B)
#   dgx-spark-uv-cache              uv package cache (~5 GB)
#   dgx-spark-apt-cache             apt cache for nsight-compute (~2 GB)
#   dgx-spark-triton-cache-{a,b,c}  Triton JIT cache, per-wheel (~500 MB each)

set -euo pipefail

VOLUMES=(
  dgx-spark-build-strict
  dgx-spark-hf-cache
  dgx-spark-uv-cache
  dgx-spark-apt-cache
  dgx-spark-triton-cache-a dgx-spark-triton-cache-b dgx-spark-triton-cache-c
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
