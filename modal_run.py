"""
Modal 云端 SWE-bench 评测脚本

使用 swerex_modal 环境在 Modal 云端运行 SWE-bench 评测。
评测结果写入 Modal Volume。

用法：
    # 单个 instance (V0 baseline)
    uv run modal run --detach modal_run.py -e main

    # V2 Prefetch 实验
    uv run modal run --detach modal_run.py -e main --config v2 --slice-spec "0:20"

    # 指定 instance 列表
    uv run modal run --detach modal_run.py -e main --instances "django__django-11039,django__django-11099"

    # 批量（slice）
    uv run modal run --detach modal_run.py -e main --slice-spec "0:5"
"""

import os
import subprocess
import json
from datetime import datetime
from pathlib import Path

import modal

# ============================================================
# Config 映射
# ============================================================
CONFIG_MAP = {
    # Baseline
    "v0": "/app/src/minisweagent/config/extra/swebench_modal_litellm.yaml",
    "baseline": "/app/src/minisweagent/config/extra/swebench_modal_litellm.yaml",
    "baseline_v0": "/app/src/minisweagent/config/extra/swebench_modal_litellm.yaml",

    # Prefetch V2 variants
    "v2": "/app/src/minisweagent/config/extra/swebench_prefetch_v2.yaml",
    "prefetch": "/app/src/minisweagent/config/extra/swebench_prefetch_v2.yaml",
    "prefetch_v2": "/app/src/minisweagent/config/extra/swebench_prefetch_v2.yaml",

    # Ablation variants
    "prefetch_v2_cache": "/app/src/minisweagent/config/extra/swebench_prefetch_v2_cache.yaml",
    "prefetch_v2_pager": "/app/src/minisweagent/config/extra/swebench_prefetch_v2_pager.yaml",
    "prefetch_v2_all": "/app/src/minisweagent/config/extra/swebench_prefetch_v2_all.yaml",
}

# ============================================================
# Modal App 配置
# ============================================================
app = modal.App("swebench-eval")

# Image 构建：使用 copy=True 后可以继续 run_commands
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "ripgrep")
    .pip_install("uv")
    # copy=True 允许后续继续 build steps
    .add_local_dir("src", remote_path="/app/src", copy=True)
    .add_local_file("pyproject.toml", remote_path="/app/pyproject.toml", copy=True)
    # 安装项目（包括 modal + swe-rex 依赖）
    .run_commands(
        "cd /app && uv pip install --system -e '.[modal]' 'swe-rex>=1.4.0'",
        "which mini-extra && mini-extra --help | head -5",
        "python -c 'from swerex.deployment.modal import ModalDeployment; print(\"swe-rex OK\")'",
    )
)

# Volume 用于存储结果
volume = modal.Volume.from_name("research-data", create_if_missing=True)

# ============================================================
# 核心评测函数
# ============================================================
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("mini-swe-secrets")],
    volumes={"/results": volume},
    timeout=3600,  # 1 hour per instance
    cpu=2.0,
    memory=4096,
)
def run_swebench_instance(
    instance_id: str,
    run_id: str,
    subset: str = "lite",
    split: str = "test",
    config_path: str = "",
) -> dict:
    """在 Modal 云端运行单个 SWE-bench instance。

    使用 swerex_modal 环境，mini-extra 会启动嵌套的 Modal Sandbox 来执行实际任务。
    """
    # 默认配置
    if not config_path:
        config_path = CONFIG_MAP["v0"]

    print("=" * 60)
    print(f"[Modal] Starting evaluation: {instance_id}")
    print(f"[Modal] Run ID: {run_id}")
    print(f"[Modal] Subset: {subset}, Split: {split}")
    print(f"[Modal] Config: {config_path}")
    print("=" * 60)

    # 环境变量检查（不泄露明文）
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "") or os.getenv("LITELLM_BASE_URL", "")
    portkey_key = os.getenv("PORTKEY_API_KEY", "")

    print(f"\n[Env] OPENAI_API_KEY: {'set (' + str(len(api_key)) + ' chars)' if api_key else 'MISSING'}")
    print(f"[Env] OPENAI_BASE_URL: {base_url if base_url else 'MISSING'}")
    print(f"[Env] PORTKEY_API_KEY: {'set' if portkey_key else 'MISSING'}")

    # 创建输出目录
    output_dir = Path(f"/results/{run_id}/{instance_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{instance_id}.traj.json"

    print(f"\n[Output] Directory: {output_dir}")
    print(f"[Output] Trajectory: {output_file}")

    # 运行 mini-extra swebench-single
    cmd = [
        "mini-extra", "swebench-single",
        "--config", config_path,
        "--subset", subset,
        "--split", split,
        "--instance", instance_id,
        "--output", str(output_file),
        "--exit-immediately",
    ]

    print(f"\n[Cmd] {' '.join(cmd)}")
    print("=" * 60)

    start_time = datetime.now()
    result_data = {
        "instance_id": instance_id,
        "run_id": run_id,
        "subset": subset,
        "split": split,
        "config": config_path,
        "start_time": start_time.isoformat(),
        "status": "running",
    }

    # Retry configuration for SandboxTimeoutError
    MAX_RETRIES = 2
    RETRY_BASE_DELAY = 30  # seconds

    for attempt in range(MAX_RETRIES + 1):
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3300,  # 55 min timeout (留 5 min buffer)
                cwd="/app",
                env={
                    **os.environ,
                    "NO_PROXY": "localhost,127.0.0.1",
                    "no_proxy": "localhost,127.0.0.1",
                }
            )

            result_data["returncode"] = proc.returncode
            result_data["stdout"] = proc.stdout[-50000:] if len(proc.stdout) > 50000 else proc.stdout
            result_data["stderr"] = proc.stderr[-10000:] if len(proc.stderr) > 10000 else proc.stderr
            result_data["attempts"] = attempt + 1

            # Check for sandbox tunnel error - retry if detected
            if "SandboxTimeoutError" in proc.stderr or "tunnels failed" in proc.stderr:
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"\n[Retry] Sandbox tunnel error detected, waiting {delay}s (attempt {attempt + 1}/{MAX_RETRIES + 1})")
                    import time
                    time.sleep(delay)
                    continue
                else:
                    result_data["status"] = "sandbox_error"
                    result_data["error"] = "SandboxTimeoutError after max retries"
                    print(f"\n[Result] SANDBOX_ERROR (max retries exceeded)")
                    break

            if proc.returncode == 0:
                result_data["status"] = "success"
                print(f"\n[Result] SUCCESS (attempt {attempt + 1})")
            else:
                result_data["status"] = "failed"
                print(f"\n[Result] FAILED (returncode={proc.returncode})")
                print(f"[stderr] {proc.stderr[:2000]}")
            break

        except subprocess.TimeoutExpired as e:
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"\n[Retry] Timeout, waiting {delay}s (attempt {attempt + 1}/{MAX_RETRIES + 1})")
                import time
                time.sleep(delay)
                continue
            result_data["status"] = "timeout"
            result_data["error"] = str(e)
            result_data["attempts"] = attempt + 1
            print(f"\n[Result] TIMEOUT (after {attempt + 1} attempts)")
            break

        except Exception as e:
            result_data["status"] = "error"
            result_data["error"] = str(e)
            result_data["attempts"] = attempt + 1
            print(f"\n[Result] ERROR: {e}")
            break

    end_time = datetime.now()
    result_data["end_time"] = end_time.isoformat()
    result_data["duration_seconds"] = (end_time - start_time).total_seconds()

    # 保存元数据
    meta_file = output_dir / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(result_data, f, indent=2)

    # 检查是否生成了 trajectory 文件
    if output_file.exists():
        result_data["trajectory_exists"] = True
        result_data["trajectory_size"] = output_file.stat().st_size
        print(f"\n[Output] Trajectory file: {output_file.stat().st_size} bytes")
    else:
        result_data["trajectory_exists"] = False
        print(f"\n[Output] WARNING: Trajectory file not created")

    # 列出输出目录
    print(f"\n[Output] Files in {output_dir}:")
    subprocess.run(["ls", "-la", str(output_dir)])

    # 提交 Volume
    volume.commit()
    print("\n[Volume] Committed.")

    return result_data


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("mini-swe-secrets")],
    volumes={"/results": volume},
    timeout=300,
)
def get_instance_ids(subset: str = "lite", split: str = "test", slice_spec: str = "") -> list[str]:
    """获取 SWE-bench instance ID 列表。"""
    from datasets import load_dataset

    dataset_mapping = {
        "full": "princeton-nlp/SWE-Bench",
        "verified": "princeton-nlp/SWE-Bench_Verified",
        "lite": "princeton-nlp/SWE-Bench_Lite",
    }

    dataset_path = dataset_mapping.get(subset, subset)
    print(f"[Dataset] Loading {dataset_path}, split={split}")

    instances = list(load_dataset(dataset_path, split=split))
    instance_ids = sorted([inst["instance_id"] for inst in instances])

    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instance_ids = instance_ids[slice(*values)]

    print(f"[Dataset] {len(instance_ids)} instances")
    return instance_ids


# ============================================================
# 本地入口
# ============================================================
@app.local_entrypoint()
def main(
    instances: str = "",
    subset: str = "lite",
    split: str = "test",
    slice_spec: str = "",
    workers: int = 2,
    config: str = "v0",
    run_tag: str = "",
):
    """运行 SWE-bench 评测。

    Args:
        instances: 逗号分隔的 instance ID 列表（如果为空则使用 slice）
        subset: 数据集子集 (lite/verified/full)
        split: 数据集分片 (test/dev)
        slice_spec: slice 规格 (如 "0:5")
        workers: 并行 worker 数量
        config: 配置版本 (v0/baseline/v2/prefetch)
        run_tag: 自定义运行标签
    """
    from datetime import datetime

    # 解析配置
    config_path = CONFIG_MAP.get(config.lower(), CONFIG_MAP["v0"])
    config_name = config.lower()

    # 生成 run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_tag:
        run_id = f"{run_tag}_{timestamp}"
    else:
        run_id = f"{config_name}_{subset}_{timestamp}"

    print(f"\n{'='*60}")
    print(f"SWE-bench Modal Evaluation")
    print(f"{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Config: {config_name} -> {config_path}")
    print(f"Subset: {subset}")
    print(f"Split: {split}")
    print(f"Slice: {slice_spec if slice_spec else 'N/A'}")
    print(f"Workers: {workers}")

    # 确定要运行的 instances
    if instances:
        instance_ids = [i.strip() for i in instances.split(",")]
        print(f"Instances (from arg): {instance_ids}")
    else:
        if not slice_spec:
            # 默认只跑 1 个 instance
            slice_spec = "0:1"
            print(f"No instances or slice specified, defaulting to slice={slice_spec}")

        print(f"\nFetching instance IDs...")
        instance_ids = get_instance_ids.remote(subset=subset, split=split, slice_spec=slice_spec)
        print(f"Instances: {instance_ids}")

    print(f"\nTotal instances to run: {len(instance_ids)}")
    print(f"{'='*60}\n")

    # 使用 map 并行运行
    # 每个调用传入相同的 run_id, subset, split, config_path
    args = [(iid, run_id, subset, split, config_path) for iid in instance_ids]

    results = []
    for result in run_swebench_instance.starmap(args, return_exceptions=True):
        if isinstance(result, Exception):
            print(f"[Error] {result}")
            results.append({"status": "exception", "error": str(result)})
        else:
            print(f"[Done] {result.get('instance_id')}: {result.get('status')} ({result.get('duration_seconds', 0):.1f}s)")
            results.append(result)

    # 汇总统计
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    status_counts = {}
    for r in results:
        s = r.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    print(f"Total: {len(results)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    print(f"\nResults saved to Modal Volume: research-data")
    print(f"Path: /results/{run_id}/")
    print(f"\nTo view results:")
    print(f"  uv run modal volume ls research-data /results/{run_id}/ -e main")

    # 保存汇总到 volume
    summary = {
        "run_id": run_id,
        "config": config_name,
        "config_path": config_path,
        "subset": subset,
        "split": split,
        "slice_spec": slice_spec,
        "total": len(results),
        "status_counts": status_counts,
        "results": results,
    }

    return summary
