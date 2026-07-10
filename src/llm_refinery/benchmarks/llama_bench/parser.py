from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

TPS_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>tok/s|tokens/s|t/s)\b", re.I)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
NUMERIC_KEY_ALIASES = {
    "t_s": "tokens_per_second",
    "tok_s": "tokens_per_second",
    "tokens_s": "tokens_per_second",
    "avg_ts": "tokens_per_second",
    "stddev_ts": "tokens_per_second_stddev",
    "samples_ts": "samples_tokens_per_second",
}


def reparse_llama_bench_run(run: dict[str, Any]) -> dict[str, float]:
    artifact = (run.get("artifacts") or {}).get("stdout")
    if not artifact:
        raise FileNotFoundError("llama-bench run has no stdout artifact")
    return parse_llama_bench_metrics(Path(artifact["path"]).read_text(encoding="utf-8"))


def parse_llama_bench_metrics(stdout: str) -> dict[str, float]:
    """Best-effort parser for llama-bench output.

    llama-bench can emit markdown tables by default and JSON when called with `-o json`.
    This parser intentionally keeps going when the exact schema changes: it extracts all
    numeric JSON leaves and common markdown `t/s` columns.
    """

    metrics: dict[str, float] = {}
    metrics.update(_parse_json_metrics(stdout))
    metrics.update(_parse_markdown_metrics(stdout))

    if not metrics:
        for index, match in enumerate(TPS_RE.finditer(stdout), start=1):
            metrics[f"tokens_per_second_{index}"] = float(match.group("value"))

    return metrics


def _parse_json_metrics(stdout: str) -> dict[str, float]:
    payload = _find_json_payload(stdout)
    if payload is None:
        return {}

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return {}

    metrics: dict[str, float] = {}
    if isinstance(data, list):
        for index, item in enumerate(data):
            prefix = _json_row_prefix(item, index)
            metrics.update(_numeric_leaves(item, prefix))
    else:
        metrics.update(_numeric_leaves(data, "json"))
    return metrics


def _find_json_payload(stdout: str) -> str | None:
    stripped = stdout.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    # Some llama.cpp builds print logs before JSON. Try the first JSON-looking span.
    candidates: list[str] = []
    for open_char, close_char in (("[", "]"), ("{", "}")):
        start = stripped.find(open_char)
        end = stripped.rfind(close_char)
        if start != -1 and end > start:
            candidates.append(stripped[start : end + 1])
    return max(candidates, key=len) if candidates else None


def _json_row_prefix(item: Any, index: int) -> str:
    if isinstance(item, dict):
        test = item.get("test") or item.get("name")
        if test:
            return _safe_name(str(test))

        n_prompt = int(item.get("n_prompt") or 0)
        n_gen = int(item.get("n_gen") or 0)
        if n_prompt and n_gen:
            return f"pp{n_prompt}_tg{n_gen}"
        if n_prompt:
            return f"pp{n_prompt}"
        if n_gen:
            return f"tg{n_gen}"

    return f"row_{index}"


def _numeric_leaves(value: Any, prefix: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(value, bool):
        return metrics
    if isinstance(value, int | float):
        metrics[prefix] = float(value)
        return metrics
    if isinstance(value, dict):
        for key, child in value.items():
            child_name = _safe_name(str(key))
            child_name = NUMERIC_KEY_ALIASES.get(child_name, child_name)
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            metrics.update(_numeric_leaves(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            metrics.update(_numeric_leaves(child, f"{prefix}.{index}"))
    return metrics


def _parse_markdown_metrics(stdout: str) -> dict[str, float]:
    rows = _markdown_rows(stdout.splitlines())
    metrics: dict[str, float] = {}
    for index, row in enumerate(rows):
        test = row.get("test") or row.get("name") or f"row_{index}"
        prefix = _safe_name(test)

        for key, value in row.items():
            normalized_key = _safe_name(key)
            number = _first_number(value)
            if number is None:
                continue

            if normalized_key in {"t_s", "tok_s", "tokens_s", "avg_ts"}:
                metrics[f"{prefix}.tokens_per_second"] = number
            else:
                metrics[f"{prefix}.{normalized_key}"] = number
    return metrics


def _markdown_rows(lines: Iterable[str]) -> list[dict[str, str]]:
    table_lines = [line.strip() for line in lines if line.strip().startswith("|")]
    if len(table_lines) < 3:
        return []

    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells:
            continue
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        if header is None:
            header = [_safe_name(cell) for cell in cells]
            continue
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells, strict=True)))
    return rows


def _first_number(value: str) -> float | None:
    match = NUMBER_RE.search(value.replace(",", ""))
    return float(match.group(0)) if match else None


def _safe_name(value: str) -> str:
    value = value.strip().lower()
    value = value.replace("/", "_").replace("%", "pct")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "value"
