"""Canonical AutoKernel keep/revert loop (arXiv:2603.21331 §4) for sm_121a kernel
synthesis. All four configs run this:
  A Learner-only  --use-mixture=False --k-expert=999999
  B Expert-only   --use-mixture=False --k-expert=0
  C Distillation  --use-mixture=False --k-expert=5
  D Full swarm    --use-mixture=True  --k-expert=5
Reward = fast_p AUC vs eager baseline. Keep if AUC improves > 3% (2× the ~1.5%
run-to-run median drift); else discard. Move-on at 5 consecutive reverts or 2× raw
speedup. State is a single in-memory best_source string — no git."""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

from .bench import bench, bench_reference
from .correctness import run_5_stage
from .data import Definition, EvaluationStatus, Workload, load_json_file
from .fast_p import fast_p_auc, median_speedup
from .ncu import flops_and_bytes, roofline_tier
from .peak import utilization
from .swarm import Swarm


def _utilization(definition, evals, tier):
    """Achieved throughput + % of peak from the median PASSED latency."""
    fb = flops_and_bytes(definition)
    lats = [e.performance.latency_ms for e in evals if e.status == EvaluationStatus.PASSED]
    return utilization(definition, fb[0], fb[1], statistics.median(lats), tier)


def _gpu_clock_mhz():
    # Provenance: the actual locked clock this run measured at. With clocks floating,
    # latency varies ~30% and all speedups are noise; results must record the lock.
    out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.applications.gr",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.splitlines()[0].strip())


def _definitions_root():
    return Path(os.environ.get("GB10_DEFINITIONS", "/opt/tune/definitions"))


def _runs_root():
    return Path(os.environ.get("GB10_RUNS", "/workspace/runs"))


def _load_workloads(definition_name):
    root = _definitions_root() / "workloads" / definition_name
    return [load_json_file(Workload, p) for p in sorted(root.glob("*.json"))]


def _partition_workloads(workloads, seed=0):
    rng = random.Random(seed)
    shuffled = list(workloads)
    rng.shuffle(shuffled)
    split = max(1, int(len(shuffled) * 0.8))
    return shuffled[:split], shuffled[split:]


def tune(definition_name, baseline_path, max_iters, k_expert, use_mixture):
    definition = load_json_file(Definition, _definitions_root() / f"{definition_name}.json")
    all_workloads = _load_workloads(definition_name)
    if not all_workloads:
        raise RuntimeError(f"no workloads under definitions/workloads/{definition_name}/")
    visible, heldout = _partition_workloads(all_workloads)

    swarm = Swarm(use_mixture=use_mixture)
    candidates_dir = _runs_root() / definition_name / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    baseline_ms_visible = bench_reference(definition, visible)
    baseline_ms_heldout = bench_reference(definition, heldout)

    best_source = baseline_path.read_text()
    best_evals = bench(definition, best_source, visible)
    best_auc_visible = fast_p_auc(best_evals, baseline_ms_visible, p_max=2.0)
    best_speedup = median_speedup(best_evals, baseline_ms_visible)
    tier = roofline_tier(definition)  # static (arithmetic intensity), constant per Definition

    asi_records = []  # GEPA ASI: {source, failure diagnostic} fed back to the proposer
    iterations = []
    n_revert = 0

    for i in range(max_iters):
        used_expert = n_revert >= k_expert
        if used_expert:
            candidate = swarm.expert_propose(definition, best_source, asi_records)
            n_revert = 0
        else:
            candidate = swarm.propose(definition, best_source, tier, asi_records)
        (candidates_dir / f"iter_{i}.py").write_text(candidate)

        ok, reason = run_5_stage(candidate, definition)
        if not ok:
            n_revert += 1
            asi_records.append({"iter": i, "source": candidate[:1000], "failure": reason})
            iterations.append({"iter": i, "decision": "revert", "reason": reason[:120], "used_expert": used_expert})
            continue

        evals = bench(definition, candidate, visible)
        auc = fast_p_auc(evals, baseline_ms_visible, p_max=2.0)

        # Keep only if clearly above the measurement floor: with clocks locked + per-iter
        # timing, run-to-run median drift is ~1.5%, so require a 3% AUC gain (2× margin).
        if auc > best_auc_visible * 1.03:
            best_source, best_auc_visible = candidate, auc
            best_speedup = median_speedup(evals, baseline_ms_visible)
            n_revert = 0
            util = _utilization(definition, evals, tier)
            iterations.append({"iter": i, "decision": "keep", "auc": auc, "speedup": best_speedup,
                               "used_expert": used_expert,
                               **({"pct_of_hw_peak": util["pct_of_hw_peak"]} if "pct_of_hw_peak" in util else {})})
        else:
            n_revert += 1
            spd = median_speedup(evals, baseline_ms_visible)
            tname = {2: "compute", 1: "memory"}.get(tier, "latency")
            asi_records.append({"iter": i, "source": candidate[:1000],
                                "failure": f"correct but not faster: auc {auc:.4f} <= best {best_auc_visible:.4f}, "
                                           f"speedup {spd:.3f} (need > best). {tname}-bound."})
            iterations.append({"iter": i, "decision": "revert", "auc": auc, "used_expert": used_expert})

        if n_revert >= 5 or best_speedup >= 2.0:
            break

    final_evals = bench(definition, best_source, heldout)
    return {
        "definition": definition_name,
        "use_mixture": use_mixture,
        "k_expert": k_expert,
        "best_auc_visible": best_auc_visible,
        "best_speedup_visible": best_speedup,
        "final_auc_heldout": fast_p_auc(final_evals, baseline_ms_heldout, p_max=2.0),
        "roofline_tier": tier,
        "gpu_clock_mhz": _gpu_clock_mhz(),  # provenance — trust numbers only if locked
        "utilization": _utilization(definition, final_evals, tier),
        "iterations": iterations,
        "final_source": best_source,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--definition", required=True)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--max-iters", type=int, default=30)
    ap.add_argument("--k-expert", type=int, default=5)
    ap.add_argument("--use-mixture", type=lambda s: s.lower() in ("1", "true", "yes"), default=True)
    args = ap.parse_args(argv)
    print(json.dumps(tune(args.definition, args.baseline, args.max_iters, args.k_expert, args.use_mixture), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
