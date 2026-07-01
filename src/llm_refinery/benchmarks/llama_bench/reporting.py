from __future__ import annotations


def metric_summary(metrics: dict[str, float], *, limit: int = 4) -> str:
    if not metrics:
        return ""

    preferred = [
        (key, value) for key, value in metrics.items() if key.endswith(".tokens_per_second")
    ]
    remaining = [(key, value) for key, value in metrics.items() if (key, value) not in preferred]
    selected = [*preferred, *remaining][:limit]
    return ", ".join(f"{key}={value:.3f}" for key, value in selected)


def tail(output: str, *, lines: int = 6) -> str:
    stripped_lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(stripped_lines[-lines:])
