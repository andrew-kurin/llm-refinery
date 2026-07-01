from __future__ import annotations

from collections.abc import Iterable

DEFAULT_PERCENTILES = (50, 90, 95, 99)


def add_distribution_metrics(
    metrics: dict[str, float],
    prefix: str,
    values: Iterable[float | int | None],
    *,
    unit_suffix: str = "_s",
    percentiles: tuple[int, ...] = DEFAULT_PERCENTILES,
) -> None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return
    metrics[f"{prefix}_avg{unit_suffix}"] = sum(clean) / len(clean)
    metrics[f"{prefix}_min{unit_suffix}"] = clean[0]
    metrics[f"{prefix}_max{unit_suffix}"] = clean[-1]
    for percentile in percentiles:
        metrics[f"{prefix}_p{percentile}{unit_suffix}"] = percentile_value(clean, percentile)


def percentile_value(sorted_values: list[float], percentile: int | float) -> float:
    if not sorted_values:
        raise ValueError("percentile_value requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    if percentile <= 1:
        percentile *= 100
    rank = (float(percentile) / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
