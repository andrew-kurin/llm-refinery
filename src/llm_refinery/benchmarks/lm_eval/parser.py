from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def reparse_lm_eval_run(run: dict[str, Any]) -> dict[str, float]:
    artifact = (run.get("artifacts") or {}).get("result")
    if not artifact:
        raise FileNotFoundError("lm-eval run has no result artifact")
    return parse_lm_eval_metrics(Path(artifact["path"]))


def latest_lm_eval_result(output_path: Path, *, newer_than: float | None = None) -> Path | None:
    if not output_path.exists():
        return None
    result_candidates = [p for p in output_path.rglob("*.json") if "result" in p.name.lower()]
    candidates = result_candidates or list(output_path.rglob("*.json"))
    if newer_than is not None:
        candidates = [path for path in candidates if path.stat().st_mtime >= newer_than]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_lm_eval_metrics(result_path: Path) -> dict[str, float]:
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    results = raw.get("results") or {}
    metrics: dict[str, float] = {}
    for task_name, task_results in results.items():
        if not isinstance(task_results, dict):
            continue
        for raw_name, value in task_results.items():
            if not isinstance(value, (int, float)):
                continue
            metric_name = normalize_lm_eval_metric_name(str(task_name), str(raw_name))
            metrics[metric_name] = float(value)
    return metrics


def normalize_lm_eval_metric_name(task_name: str, raw_name: str) -> str:
    if "," not in raw_name:
        return f"{task_name}.{raw_name}"
    metric, filter_name = raw_name.split(",", 1)
    return f"{task_name}.{filter_name}.{metric}"
