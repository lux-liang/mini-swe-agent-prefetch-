"""Run on a single SWE-Bench instance."""

import traceback
from pathlib import Path

import typer
import yaml
from datasets import load_dataset

from minisweagent import global_config_dir
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.agents.prefetch_agent import PrefetchAgent  # P3: 添加 PrefetchAgent 导入
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.models import get_model
from minisweagent.run.extra.swebench import (
    DATASET_MAPPING,
    get_sb_environment,
)
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import logger

app = typer.Typer(add_completion=False)

DEFAULT_OUTPUT = global_config_dir / "last_swebench_single_run.traj.json"


# fmt: off
@app.command()
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    instance_spec: str = typer.Option(0, "-i", "--instance", help="SWE-Bench instance ID or index", rich_help_panel="Data selection"),
    model_name: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "-c", "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    config_path: Path = typer.Option( builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", rich_help_panel="Advanced"),
    exit_immediately: bool = typer.Option( False, "--exit-immediately", help="Exit immediately when the agent wants to finish instead of prompting.", rich_help_panel="Basic"),
    output: Path = typer.Option(DEFAULT_OUTPUT, "-o", "--output", help="Output trajectory file", rich_help_panel="Basic"),
) -> None:
    # fmt: on
    """Run on a single SWE-Bench instance."""
    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset from {dataset_path}, split {split}...")
    instances = {
        inst["instance_id"]: inst  # type: ignore
        for inst in load_dataset(dataset_path, split=split)
    }
    if instance_spec.isnumeric():
        instance_spec = sorted(instances.keys())[int(instance_spec)]
    instance: dict = instances[instance_spec]  # type: ignore

    config_path = get_config_path(config_path)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    if exit_immediately:
        config.setdefault("agent", {})["confirm_exit"] = False
    env = get_sb_environment(config, instance)

    # ========== P3: Prefetch-aware agent selection ==========
    prefetch_cfg = config.get("prefetch") or {}
    prefetch_enabled = bool(prefetch_cfg.get("enabled", False))
    task = instance["problem_statement"]

    if prefetch_enabled:
        agent = PrefetchAgent(
            get_model(model_name, config.get("model", {})),
            env,
            instance_id=instance_spec,
            bug_anchor_text=task[:500],
            progress_manager=None,  # single mode 无 progress_manager
            prefetch_enabled=True,
            prefetch_base_budget=prefetch_cfg.get("base_budget", 4000),
            prefetch_verified_mode=prefetch_cfg.get("verified_mode", True),
            prefetch_max_pager_blocks=prefetch_cfg.get("max_pager_blocks", 2),
            prefetch_evidence_top_k=prefetch_cfg.get("evidence_top_k", 2),
            prefetch_shadow_top_m=prefetch_cfg.get("shadow_top_m", 5),
            **({"mode": "yolo"} | config.get("agent", {})),
        )
    else:
        agent = InteractiveAgent(
            get_model(model_name, config.get("model", {})),
            env,
            **({"mode": "yolo"} | config.get("agent", {})),
        )
    # ========== End P3 ==========

    exit_status, result, extra_info = None, None, None
    try:
        exit_status, result = agent.run(task)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error processing instance {instance_spec}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        save_traj(agent, output, exit_status=exit_status, result=result, extra_info=extra_info)  # type: ignore[arg-type]


if __name__ == "__main__":
    app()
