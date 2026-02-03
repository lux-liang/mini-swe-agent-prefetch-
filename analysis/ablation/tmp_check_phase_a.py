import os
import json
from collections import Counter, defaultdict
from pathlib import Path

RUNS = [
    "phase_a_prefetch_v2_all_verified_20_40_20260129_171516",
    "phase_a_prefetch_v2_all_verified_280_300_20260129_171619",
    "phase_a_v0_verified_20_40_20260129_171448",
    "phase_a_v0_verified_280_300_20260129_171549",
]
BASE = Path("results/phase_a_downloaded")

# Basic keyword bucketing for error messages
KEYWORDS = [
    "timeout",
    "timed out",
    "oom",
    "out of memory",
    "cuda",
    "gpu",
    "modal",
    "volume",
    "network",
    "connection",
    "http",
    "rate limit",
    "ratelimit",
    "quota",
    "disk",
    "no space",
    "killed",
    "sigterm",
    "sigkill",
    "kube",
    "preempt",
    "crash",
    "error",
    "exception",
]


def bucket_error(text: str) -> str:
    t = (text or "").lower()
    for kw in KEYWORDS:
        if kw in t:
            return kw
    return "(other/unknown)"


def load_meta(meta_path: Path):
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return {}


def extract_status(meta: dict) -> str:
    for k in ["status", "result", "outcome"]:
        if k in meta:
            v = meta.get(k)
            if isinstance(v, dict) and "status" in v:
                return str(v.get("status"))
            return str(v)
    return "unknown"


def extract_error_text(meta: dict) -> str:
    # Try common locations
    for k in ["error", "exception", "stderr", "traceback", "message", "failure_reason"]:
        if k in meta:
            v = meta.get(k)
            if isinstance(v, dict):
                return json.dumps(v)
            return str(v)
    # Fallback: search nested strings
    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield from walk(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from walk(v)
        elif isinstance(obj, str):
            yield obj
    texts = list(walk(meta))
    return " ".join(texts[:20]) if texts else ""


for run in RUNS:
    run_path = BASE / run
    if not run_path.exists():
        print(f"RUN {run} NOT FOUND")
        continue

    instances = [p for p in run_path.iterdir() if p.is_dir() and "__" in p.name]
    total = len(instances)
    with_traj = []
    only_meta = []

    for inst in instances:
        has_traj = any(f.name.endswith('.traj.json') for f in inst.iterdir())
        if has_traj:
            with_traj.append(inst)
        else:
            only_meta.append(inst)

    success_no_traj = 0
    status_counts = Counter()
    bucket_counts = Counter()

    for inst in only_meta:
        meta_path = inst / "meta.json"
        meta = load_meta(meta_path)
        status = extract_status(meta).lower()
        status_counts[status] += 1
        if status == "success":
            success_no_traj += 1
        err_text = extract_error_text(meta)
        bucket = bucket_error(err_text)
        bucket_counts[bucket] += 1

    print("\nRUN", run)
    print("  total_instances:", total)
    print("  with_traj:", len(with_traj))
    print("  only_meta:", len(only_meta))
    print("  only_meta_status_counts:", dict(status_counts))
    print("  only_meta_success_count:", success_no_traj)
    print("  error_buckets_top15:")
    for k, v in bucket_counts.most_common(15):
        print(f"    {k}: {v}")
