from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_refinery.core.runs import stable_hash


@dataclass(frozen=True)
class ParsedLmEvalSample:
    sample_id: str
    task_name: str
    payload: dict[str, Any]
    metrics: dict[str, float]


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
    return add_lm_eval_confidence_intervals(metrics)


def normalize_lm_eval_metric_name(task_name: str, raw_name: str) -> str:
    if "," not in raw_name:
        return f"{task_name}.{raw_name}"
    metric, filter_name = raw_name.split(",", 1)
    return f"{task_name}.{filter_name}.{metric}"


def add_lm_eval_confidence_intervals(metrics: dict[str, float]) -> dict[str, float]:
    """Add approximate 95% intervals when lm-eval reports a standard error."""
    enriched = dict(metrics)
    for stderr_name, stderr in metrics.items():
        if not stderr_name.endswith("_stderr") or not math.isfinite(stderr) or stderr < 0:
            continue
        metric_name = stderr_name.removesuffix("_stderr")
        estimate = metrics.get(metric_name)
        if estimate is None or not math.isfinite(estimate):
            continue
        margin = 1.96 * stderr
        lower = estimate - margin
        upper = estimate + margin
        if 0.0 <= estimate <= 1.0:
            lower = max(0.0, lower)
            upper = min(1.0, upper)
        enriched[f"{metric_name}_ci95_low"] = lower
        enriched[f"{metric_name}_ci95_high"] = upper
    return enriched


def lm_eval_sample_files(result_path: Path) -> list[Path]:
    """Find the JSONL sample artifacts emitted alongside one lm-eval result."""
    if not result_path.name.startswith("results_") or result_path.suffix != ".json":
        return []
    suffix = result_path.stem.removeprefix("results_")
    return sorted(result_path.parent.glob(f"samples_*_{suffix}.jsonl"))


def parse_lm_eval_samples(
    sample_path: Path,
    *,
    result_path: Path,
) -> list[ParsedLmEvalSample]:
    task_name = _sample_task_name(sample_path, result_path=result_path)
    samples: list[ParsedLmEvalSample] = []
    seen_ids: dict[str, int] = {}
    with sample_path.open(encoding="utf-8") as sample_file:
        for line_number, line in enumerate(sample_file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid lm-eval sample JSON in {sample_path.name}:{line_number}: {exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(
                    f"lm-eval sample in {sample_path.name}:{line_number} is not an object"
                )
            filter_name = str(raw.get("filter") or "default")
            doc_id = str(raw.get("doc_id", line_number - 1))
            base_id = f"{task_name}:{filter_name}:{doc_id}"
            duplicate_index = seen_ids.get(base_id, 0)
            seen_ids[base_id] = duplicate_index + 1
            sample_id = base_id if duplicate_index == 0 else f"{base_id}:{duplicate_index}"
            metrics = _sample_metrics(raw)
            correct = _sample_correctness(metrics, task_name=task_name)
            if correct is not None:
                metrics["correct"] = correct
            response = raw.get("filtered_resps", raw.get("resps"))
            samples.append(
                ParsedLmEvalSample(
                    sample_id=sample_id,
                    task_name=task_name,
                    payload={
                        "task": task_name,
                        "doc_id": raw.get("doc_id", line_number - 1),
                        "filter": filter_name,
                        "doc_hash": raw.get("doc_hash"),
                        "prompt_hash": raw.get("prompt_hash"),
                        "target_hash": raw.get("target_hash"),
                        "response_hash": stable_hash(response),
                        "sample_artifact": sample_path.name,
                        "sample_line": line_number,
                    },
                    metrics=metrics,
                )
            )
    return samples


def summarize_lm_eval_samples(
    samples: list[ParsedLmEvalSample],
) -> dict[str, float]:
    metrics: dict[str, float] = {"samples.recorded_count": float(len(samples))}
    task_names = sorted({sample.task_name for sample in samples})
    for task_name in task_names:
        task_samples = [sample for sample in samples if sample.task_name == task_name]
        correct_values = [
            sample.metrics["correct"] for sample in task_samples if "correct" in sample.metrics
        ]
        prefix = f"samples.{task_name}"
        metrics[f"{prefix}.recorded_count"] = float(len(task_samples))
        if not correct_values:
            continue
        correct_count = sum(correct_values)
        sample_count = len(correct_values)
        correct_rate = correct_count / sample_count
        lower, upper = _wilson_interval(correct_count, sample_count)
        metrics[f"{prefix}.scored_count"] = float(sample_count)
        metrics[f"{prefix}.correct_count"] = float(correct_count)
        metrics[f"{prefix}.correct_rate"] = correct_rate
        metrics[f"{prefix}.correct_rate_ci95_low"] = lower
        metrics[f"{prefix}.correct_rate_ci95_high"] = upper
    return metrics


def _sample_task_name(sample_path: Path, *, result_path: Path) -> str:
    result_suffix = result_path.stem.removeprefix("results_")
    name = sample_path.stem.removeprefix("samples_")
    suffix = f"_{result_suffix}"
    if not name.endswith(suffix):
        raise ValueError(
            f"sample artifact {sample_path.name} does not match result {result_path.name}"
        )
    task_name = name[: -len(suffix)]
    if not task_name:
        raise ValueError(f"cannot infer task name from lm-eval sample artifact {sample_path.name}")
    return task_name


def _sample_metrics(raw: dict[str, Any]) -> dict[str, float]:
    excluded = {"doc_id", "arguments"}
    return {
        str(name): float(value)
        for name, value in raw.items()
        if name not in excluded
        and isinstance(value, (bool, int, float))
        and math.isfinite(float(value))
    }


def _sample_correctness(metrics: dict[str, float], *, task_name: str) -> float | None:
    preferred = (
        ("prompt_level_loose_acc", "prompt_level_strict_acc")
        if task_name == "ifbench"
        else ("prompt_level_strict_acc", "prompt_level_loose_acc")
    ) + (
        "prompt_strict_acc",
        "exact_match",
        "acc_norm",
        "acc",
        "em",
    )
    for name in preferred:
        if name in metrics:
            return 1.0 if metrics[name] > 0 else 0.0
    return None


def _wilson_interval(successes: float, sample_count: int) -> tuple[float, float]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    z = 1.96
    proportion = successes / sample_count
    denominator = 1.0 + z**2 / sample_count
    center = (proportion + z**2 / (2 * sample_count)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / sample_count + z**2 / (4 * sample_count**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
