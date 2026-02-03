#!/usr/bin/env python3
import os, re, json, csv, math
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("analysis/ablation/phase_a_final")
PLOTS_DIR = OUTPUT_DIR / "plots"

SLICES = ["0_20", "20_40", "280_300"]
GROUP_V0 = "v0"
GROUP_PREFETCH = "prefetch_v2_all"

EXPLORE_KW = ["rg","grep","ls","cat","nl","sed -n","head","tail","read_file","find"]
NON_EXPLORE_KW = ["apply_patch","edit","write_file","pytest","python","submit","run_tests","mvn","npm","make"]

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}

def detect_base_dir():
    merged = Path("results/phase_a_merged")
    downloaded = Path("results/phase_a_downloaded")
    if merged.exists(): return merged, "merged"
    if downloaded.exists(): return downloaded, "downloaded"
    raise FileNotFoundError("No results directory found (results/phase_a_merged or results/phase_a_downloaded).")

def find_dirs_by_prefix(base, prefix):
    return sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)])

def pick_single_dir(base, prefix):
    cand = find_dirs_by_prefix(base, prefix)
    return cand[-1] if cand else None

def build_source_map(base, mode):
    src = {s: {GROUP_V0: [], GROUP_PREFETCH: []} for s in SLICES}
    if mode == "merged":
        for s in SLICES:
            v0_dir = base / f"v0_{s}"
            pf_dir = base / f"{GROUP_PREFETCH}_{s}"
            if v0_dir.exists(): src[s][GROUP_V0].append(v0_dir)
            if pf_dir.exists(): src[s][GROUP_PREFETCH].append(pf_dir)
        return src

    v0_map = {
        "0_20": "phase_a_v0_verified_0_20_20260129_171349",
        "20_40": "phase_a_v0_verified_20_40_20260129_171448",
        "280_300": "phase_a_v0_verified_280_300_20260129_171549",
    }
    pf_map = {
        "0_20": "phase_a_prefetch_v2_all_verified_0_20_20260129_171416",
        "20_40": "phase_a_prefetch_v2_all_verified_20_40_20260129_171516",
        "280_300": "phase_a_prefetch_v2_all_verified_280_300_20260129_171619",
    }
    for s in SLICES:
        if (base / v0_map[s]).exists(): src[s][GROUP_V0].append(base / v0_map[s])
        if (base / pf_map[s]).exists(): src[s][GROUP_PREFETCH].append(base / pf_map[s])

    for r in find_dirs_by_prefix(base, "phase_a_prefetch_repair_20_40_b"):
        src["20_40"][GROUP_PREFETCH].append(r)

    v0_rep = pick_single_dir(base, "phase_a_v0_repair_20_40_")
    if v0_rep: src["20_40"][GROUP_V0].append(v0_rep)
    return src

def extract_commands_from_message(content):
    if not content: return []
    cmds = []
    for m in re.findall(r"```bash\s*\n(.*?)\n```", content, re.DOTALL):
        cmds.extend([ln.strip() for ln in m.splitlines() if ln.strip()])
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("$"):
            cmds.append(line[1:].strip())
    return cmds

def is_explore(cmd_text):
    t = cmd_text.lower()
    if any(k in t for k in NON_EXPLORE_KW): return False
    return any(k in t for k in EXPLORE_KW)

def parse_paging_cmd(cmd):
    cmd = cmd.strip()
    m = re.search(r"nl\s+-ba\s+(\S+)\s*\|\s*sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?", cmd)
    if m: return m.group(1), int(m.group(2)), int(m.group(3))
    m = re.search(r"sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s+(\S+)", cmd)
    if m: return m.group(3), int(m.group(1)), int(m.group(2))
    m = re.search(r"head\s+-n\s+(\d+)\s+(\S+)\s*\|\s*tail\s+-n\s+(\d+)", cmd)
    if m:
        head_n = int(m.group(1)); tail_n = int(m.group(3))
        end = head_n; start = max(1, head_n - tail_n + 1)
        return m.group(2), start, end
    return None

def compute_metrics(traj, meta):
    messages = traj.get("messages", []) or []
    info = traj.get("info", {}) or {}

    if isinstance(traj.get("steps"), list):
        n_steps = len(traj["steps"])
    else:
        n_steps = sum(1 for m in messages if m.get("role") == "assistant")

    commands = []
    for m in messages:
        if m.get("role") == "assistant":
            commands.extend(extract_commands_from_message(m.get("content","")))

    cost = None
    ms = (info.get("model_stats", {}) or {})
    if isinstance(ms.get("instance_cost"), (int, float)):
        cost = ms.get("instance_cost")

    status = (meta.get("status") or meta.get("exit_status") or "").lower()
    success = (status == "success") or (meta.get("success") is True) or (info.get("success") is True) or (info.get("resolved") is True)

    paging_ops = 0
    total_lines = 0
    spans_by_file = defaultdict(list)

    for cmd in commands:
        parsed = parse_paging_cmd(cmd)
        if parsed:
            f,s,e = parsed
            paging_ops += 1
            total_lines += max(0, e - s + 1)
            spans_by_file[f].append((s,e))

    unique_lines = 0
    for f, spans in spans_by_file.items():
        covered = set()
        for s,e in spans:
            covered.update(range(s, e+1))
        unique_lines += len(covered)

    read_dup_rate = 0.0
    if total_lines > 0:
        read_dup_rate = 1.0 - (unique_lines / total_lines)

    explore_flags = []
    for m in messages:
        if m.get("role") != "assistant": continue
        cmds = extract_commands_from_message(m.get("content",""))
        explore_flags.append(any(is_explore(c) for c in cmds) if cmds else False)

    front_n = max(1, math.ceil(len(explore_flags) * 0.25))
    front = explore_flags[:front_n]
    front_ratio = (sum(front) / len(front)) if front else 0.0

    max_chain = 0; cur = 0
    for x in explore_flags:
        if x: cur += 1; max_chain = max(max_chain, cur)
        else: cur = 0

    top_file_share = 0.0
    if paging_ops > 0:
        cnt = Counter({f: len(spans) for f,spans in spans_by_file.items()})
        top_file_share = max(cnt.values()) / paging_ops if cnt else 0.0

    return {
        "n_steps": int(n_steps),
        "cost": cost,
        "success": bool(success),
        "manual_paging_ops": int(paging_ops),
        "unique_lines": int(unique_lines),
        "total_lines": int(total_lines),
        "read_duplication_rate": float(read_dup_rate),
        "front_25_explore_ratio": float(front_ratio),
        "explore_chain_len": int(max_chain),
        "top_file_share": float(top_file_share),
        "has_prefetch_aggregate": ("prefetch_aggregate" in traj),
        "meta_status": status or "unknown",
    }

def collect_instances(run_dirs):
    traj_map = {}
    meta_map = {}
    meta_only = set()
    for run_dir in run_dirs:
        for inst_dir in run_dir.iterdir():
            if not inst_dir.is_dir(): continue
            if "__" not in inst_dir.name: continue
            meta = {}
            meta_path = inst_dir / "meta.json"
            if meta_path.exists():
                meta = load_json(meta_path)
            traj_files = list(inst_dir.glob("*.traj.json"))
            if traj_files:
                traj = load_json(traj_files[0])
                traj_map[inst_dir.name] = traj
                meta_map[inst_dir.name] = meta
            else:
                meta_only.add(inst_dir.name)
                meta_map[inst_dir.name] = meta
    return traj_map, meta_map, meta_only

def plot_box(data_by_group, title, path):
    fig, ax = plt.subplots(figsize=(6,4))
    labels = list(data_by_group.keys())
    data = [data_by_group[k] for k in labels]
    ax.boxplot(data, labels=labels)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_hist(values, title, path):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(values, bins=20)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_scatter(x, y, title, xlabel, ylabel, path):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(x, y, s=10, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def median(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return None
    xs = sorted(xs)
    n = len(xs)
    if n % 2 == 1: return xs[n//2]
    return 0.5 * (xs[n//2 - 1] + xs[n//2])

def main():
    ensure_dirs()
    base, mode = detect_base_dir()
    src = build_source_map(base, mode)

    summary = {"generated_at": datetime.utcnow().isoformat()+"Z", "mode": mode, "slices": {}, "sanity_checks": {}, "cost_missing": False}
    metrics_rows = []
    paired_rows = []

    s1_violation = 0
    s2_missing = 0
    s2_total = 0

    per_slice_deltas = {s: {"steps": [], "paging": [], "dup": []} for s in SLICES}

    for s in SLICES:
        v0_dirs = src[s][GROUP_V0]
        pf_dirs = src[s][GROUP_PREFETCH]

        v0_traj, v0_meta, v0_meta_only = collect_instances(v0_dirs)
        pf_traj, pf_meta, pf_meta_only = collect_instances(pf_dirs)

        v0_metrics = {}
        pf_metrics = {}

        for iid, traj in v0_traj.items():
            m = compute_metrics(traj, v0_meta.get(iid, {}))
            if m["has_prefetch_aggregate"]:
                s1_violation += 1
            v0_metrics[iid] = m
            metrics_rows.append({"slice": s, "group": GROUP_V0, "instance_id": iid, **m})

        for iid, traj in pf_traj.items():
            m = compute_metrics(traj, pf_meta.get(iid, {}))
            if not m["has_prefetch_aggregate"]:
                s2_missing += 1
            s2_total += 1
            pf_metrics[iid] = m
            metrics_rows.append({"slice": s, "group": GROUP_PREFETCH, "instance_id": iid, **m})

        paired_ids = sorted(set(v0_metrics) & set(pf_metrics))
        for iid in paired_ids:
            v0m = v0_metrics[iid]; pfm = pf_metrics[iid]
            ds = pfm["n_steps"] - v0m["n_steps"]
            dp = pfm["manual_paging_ops"] - v0m["manual_paging_ops"]
            dd = pfm["read_duplication_rate"] - v0m["read_duplication_rate"]
            per_slice_deltas[s]["steps"].append(ds)
            per_slice_deltas[s]["paging"].append(dp)
            per_slice_deltas[s]["dup"].append(dd)
            paired_rows.append({
                "slice": s, "instance_id": iid,
                "steps_delta": ds,
                "paging_ops_delta": dp,
                "read_dup_delta": dd,
                "front_25_explore_delta": pfm["front_25_explore_ratio"] - v0m["front_25_explore_ratio"],
                "cost_delta": None if (v0m["cost"] is None or pfm["cost"] is None) else (pfm["cost"] - v0m["cost"]),
            })

        summary["slices"][s] = {
            "total_instances": len(set(list(v0_traj)+list(pf_traj)+list(v0_meta_only)+list(pf_meta_only))),
            "with_traj_v0": len(v0_traj),
            "with_traj_prefetch": len(pf_traj),
            "meta_only_v0": len(v0_meta_only),
            "meta_only_prefetch": len(pf_meta_only),
            "paired_n": len(paired_ids),
        }

        def values(metrics, key):
            return [m[key] for m in metrics.values() if m.get(key) is not None]

        plot_box({GROUP_V0: values(v0_metrics,"n_steps"), GROUP_PREFETCH: values(pf_metrics,"n_steps")},
                 f"steps_distribution_{s}", PLOTS_DIR / f"steps_distribution_{s}.png")
        plot_box({GROUP_V0: values(v0_metrics,"manual_paging_ops"), GROUP_PREFETCH: values(pf_metrics,"manual_paging_ops")},
                 f"paging_ops_distribution_{s}", PLOTS_DIR / f"paging_ops_distribution_{s}.png")
        plot_box({GROUP_V0: values(v0_metrics,"front_25_explore_ratio"), GROUP_PREFETCH: values(pf_metrics,"front_25_explore_ratio")},
                 f"front_25_explore_ratio_{s}", PLOTS_DIR / f"front_25_explore_ratio_{s}.png")
        plot_box({GROUP_V0: values(v0_metrics,"read_duplication_rate"), GROUP_PREFETCH: values(pf_metrics,"read_duplication_rate")},
                 f"read_duplication_rate_{s}", PLOTS_DIR / f"read_duplication_rate_{s}.png")

        if any(m["cost"] is not None for m in v0_metrics.values()) and any(m["cost"] is not None for m in pf_metrics.values()):
            plot_scatter(values(v0_metrics,"n_steps")+values(pf_metrics,"n_steps"),
                         values(v0_metrics,"cost")+values(pf_metrics,"cost"),
                         f"cost_vs_steps_{s}", "steps", "cost", PLOTS_DIR / f"cost_vs_steps_{s}.png")
        else:
            summary["cost_missing"] = True

        plot_hist(per_slice_deltas[s]["steps"], f"paired_deltas_steps_{s}", PLOTS_DIR / f"paired_deltas_steps_{s}.png")
        plot_hist(per_slice_deltas[s]["paging"], f"paired_deltas_paging_{s}", PLOTS_DIR / f"paired_deltas_paging_{s}.png")

    missing_ratio = (s2_missing / s2_total) if s2_total else 0.0
    summary["sanity_checks"]["S1_baseline_has_prefetch_aggregate_count"] = s1_violation
    summary["sanity_checks"]["S2_prefetch_missing_prefetch_aggregate_count"] = s2_missing
    summary["sanity_checks"]["S2_prefetch_missing_ratio"] = missing_ratio

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "SUMMARY.json").write_text(json.dumps(summary, indent=2))

    def slice_line(s):
        pn = summary["slices"][s]["paired_n"]
        ms = median(per_slice_deltas[s]["steps"])
        mp = median(per_slice_deltas[s]["paging"])
        md = median(per_slice_deltas[s]["dup"])
        return f"{s} paired_n={pn} steps_delta_median={ms} paging_ops_delta_median={mp} read_dup_delta_median={md}"

    print(slice_line("0_20"))
    print(slice_line("20_40"))
    print(slice_line("280_300"))
    print(f"S1_baseline_prefetch_agg={s1_violation}")
    print(f"S2_prefetch_missing_ratio={missing_ratio:.6f}")
    print(f"cost_missing={summary['cost_missing']}")

if __name__ == "__main__":
    main()
