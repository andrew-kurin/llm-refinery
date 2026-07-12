from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PairedQualityComparison:
    paired_count: int
    baseline_only_count: int
    candidate_only_count: int
    baseline_correct_count: int
    candidate_correct_count: int
    candidate_win_count: int
    candidate_loss_count: int
    tie_count: int
    baseline_accuracy: float
    candidate_accuracy: float
    accuracy_delta: float
    accuracy_delta_ci95_low: float
    accuracy_delta_ci95_high: float
    mcnemar_exact_p: float


def compare_paired_correctness(
    baseline_samples: list[dict[str, Any]],
    candidate_samples: list[dict[str, Any]],
    *,
    task: str | None = None,
    sample_metric: str = "correct",
) -> PairedQualityComparison:
    baseline = _correctness_by_id(baseline_samples, task=task, sample_metric=sample_metric)
    candidate = _correctness_by_id(candidate_samples, task=task, sample_metric=sample_metric)
    paired_ids = sorted(baseline.keys() & candidate.keys())
    if not paired_ids:
        task_detail = f" for task {task!r}" if task else ""
        raise ValueError(f"runs have no paired correctness samples{task_detail}")

    baseline_correct = sum(baseline[sample_id] for sample_id in paired_ids)
    candidate_correct = sum(candidate[sample_id] for sample_id in paired_ids)
    candidate_wins = sum(
        1 for sample_id in paired_ids if baseline[sample_id] == 0 and candidate[sample_id] == 1
    )
    candidate_losses = sum(
        1 for sample_id in paired_ids if baseline[sample_id] == 1 and candidate[sample_id] == 0
    )
    paired_count = len(paired_ids)
    delta = (candidate_correct - baseline_correct) / paired_count
    interval_low, interval_high = _paired_normal_interval(
        paired_count=paired_count,
        wins=candidate_wins,
        losses=candidate_losses,
        delta=delta,
    )
    return PairedQualityComparison(
        paired_count=paired_count,
        baseline_only_count=len(baseline.keys() - candidate.keys()),
        candidate_only_count=len(candidate.keys() - baseline.keys()),
        baseline_correct_count=baseline_correct,
        candidate_correct_count=candidate_correct,
        candidate_win_count=candidate_wins,
        candidate_loss_count=candidate_losses,
        tie_count=paired_count - candidate_wins - candidate_losses,
        baseline_accuracy=baseline_correct / paired_count,
        candidate_accuracy=candidate_correct / paired_count,
        accuracy_delta=delta,
        accuracy_delta_ci95_low=interval_low,
        accuracy_delta_ci95_high=interval_high,
        mcnemar_exact_p=_mcnemar_exact_p(candidate_wins, candidate_losses),
    )


def _correctness_by_id(
    samples: list[dict[str, Any]], *, task: str | None, sample_metric: str
) -> dict[str, int]:
    correctness: dict[str, int] = {}
    for sample in samples:
        payload = sample.get("payload_json") or {}
        if task is not None and payload.get("task") != task:
            continue
        raw_correct = (sample.get("metrics") or {}).get(sample_metric)
        if not isinstance(raw_correct, (bool, int, float)) or float(raw_correct) not in (0.0, 1.0):
            continue
        correctness[str(sample["sample_id"])] = int(bool(raw_correct))
    return correctness


def _paired_normal_interval(
    *, paired_count: int, wins: int, losses: int, delta: float
) -> tuple[float, float]:
    if paired_count <= 1:
        return delta, delta
    sum_squares = wins + losses
    sample_variance = (sum_squares - paired_count * delta**2) / (paired_count - 1)
    standard_error = math.sqrt(max(0.0, sample_variance) / paired_count)
    margin = 1.96 * standard_error
    return max(-1.0, delta - margin), min(1.0, delta + margin)


def _mcnemar_exact_p(wins: int, losses: int) -> float:
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = min(wins, losses)
    log_probabilities = [
        math.lgamma(discordant + 1)
        - math.lgamma(successes + 1)
        - math.lgamma(discordant - successes + 1)
        - discordant * math.log(2.0)
        for successes in range(tail + 1)
    ]
    maximum = max(log_probabilities)
    one_sided = math.exp(maximum) * sum(
        math.exp(log_probability - maximum) for log_probability in log_probabilities
    )
    return min(1.0, 2.0 * one_sided)
