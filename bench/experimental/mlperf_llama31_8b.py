"""
Tier 4: MLPerf inference v5.1 llama3.1-8b wrapper.

llama3.1-8b was introduced in MLPerf inference v5.1 (Sept 2025); v5.0 only
includes llama3.1-405B + Llama-2-70B-Interactive. Plan targets v5.1, verified
against mlcommons.org/2025/09/small-llm-inference-5-1/.

This script doesn't replicate the MLPerf harness — it wraps `mlcr` (MLCommons
Runner), the canonical MLPerf v5.1 entrypoint, inside the MLCommons reference
container ghcr.io/mlcommons/inference:5.1-dev.

Standalone invocation:
  bash bench/experimental/run_e2e.sh                              # Tier 4 default
  python bench/experimental/mlperf_llama31_8b.py --scenario Offline --duration 60

Requires (host env):
  - BENCH_DOWNLOAD_MODELS=1  (gate: implies user has accepted the 16 GB download)
  - HF_TOKEN set (or HF cache mounted) — Llama 3.1 is gated under Meta's
    Community License at huggingface.co/meta-llama/Llama-3.1-8B-Instruct.

Output: bench/logs/mlperf_llama31_8b.json with tokens/s, TTFT, ITL p50/p99.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


MLPERF_IMAGE = "ghcr.io/mlcommons/inference:5.1-dev"
MLPERF_VOLUME = "mlperf-cache"
HF_CACHE_HOST = Path(os.path.expanduser("~/.cache/huggingface"))


def have_docker() -> bool:
    return shutil.which("docker") is not None


def host_hf_token() -> str | None:
    """Pull HF token from the standard hf 1.x cache location."""
    token_path = HF_CACHE_HOST / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return os.environ.get("HF_TOKEN")


def run_mlperf(scenario: str, duration_s: int, out_json: Path) -> int:
    if os.environ.get("BENCH_DOWNLOAD_MODELS") != "1":
        print("[mlperf] refused: set BENCH_DOWNLOAD_MODELS=1 to confirm "
              "16 GB Llama 3.1 8B download from HuggingFace", file=sys.stderr)
        return 2

    token = host_hf_token()
    if not token:
        print("[mlperf] FATAL: no HF token (not in ~/.cache/huggingface/token "
              "or $HF_TOKEN); Llama 3.1 8B is gated", file=sys.stderr)
        return 2

    # Ensure cache volume exists (idempotent)
    subprocess.run(["docker", "volume", "create", MLPERF_VOLUME],
                   stdout=subprocess.DEVNULL, check=True)

    out_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--ipc=host", "--shm-size=8g",
        "-e", f"HF_TOKEN={token}",
        "-v", f"{REPO}:/repo",
        "-v", f"{MLPERF_VOLUME}:/root/.cache/mlperf",
        "-v", f"{HF_CACHE_HOST}:/root/.cache/huggingface:ro",
        MLPERF_IMAGE,
        "bash", "-c",
        # mlcr is MLCommons' v5.1 unified runner; --quiet keeps log noise down
        f"set -e; mlcr run-mlperf,inference,_full,_r5.1-dev "
        f"--model=llama3_1-8b --backend=vllm --scenario={scenario} "
        f"--duration={duration_s} --output_dir=/repo/bench/logs/mlperf "
        f"--quiet 2>&1 | tee /repo/bench/logs/mlperf/run.log",
    ]
    print(f"[mlperf] launching: {' '.join(cmd[:3])} ...", file=sys.stderr)
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        print(f"[mlperf] non-zero exit {res.returncode}", file=sys.stderr)
        return res.returncode

    # MLPerf writes summary.json under output_dir/summary.json — parse it
    summary = REPO / "bench" / "logs" / "mlperf" / "summary.json"
    if summary.exists():
        payload = json.loads(summary.read_text())
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"[mlperf] wrote {out_json}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="Offline",
                    choices=["Offline", "Server", "Interactive"])
    ap.add_argument("--duration", type=int, default=60, help="seconds")
    ap.add_argument("--out", default=str(REPO / "bench/logs/mlperf_llama31_8b.json"))
    args = ap.parse_args()
    if not have_docker():
        print("[mlperf] FATAL: docker not in PATH", file=sys.stderr)
        return 2
    return run_mlperf(args.scenario, args.duration, Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
