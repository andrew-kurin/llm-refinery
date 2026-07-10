from __future__ import annotations

import re
from typing import Any

DEFAULT_METRICS = ("pp_tps", "tg_tps")
DEFAULT_SORT = "tg_tps"
ALWAYS_PARAMS = ("prompt_tokens", "gen_tokens")
IDENTITY_COLUMNS = ("model", *ALWAYS_PARAMS)


class CompareError(ValueError):
    """Raised when comparison options are invalid."""


def build_compare_rows(
    runs: list[dict[str, Any]],
    *,
    metrics: tuple[str, ...] = (),
    params: tuple[str, ...] = (),
    sort_key: str | None = None,
    ascending: bool = False,
    limit: int = 20,
    dedupe_configs: bool = True,
) -> list[dict[str, Any]]:
    metric_keys = metrics or DEFAULT_METRICS
    param_keys = params or infer_param_keys(runs)
    sort_key = sort_key or (metric_keys[0] if metrics else DEFAULT_SORT)

    rows = [_build_compare_row(run, metric_keys=metric_keys, param_keys=param_keys) for run in runs]
    if dedupe_configs:
        rows = _dedupe_latest_configs(rows, param_keys=param_keys)
    rows.sort(key=lambda row: _sort_key(row.get(sort_key), ascending=ascending))
    return rows[:limit]


def build_compare_table_rows(rows: list[dict[str, Any]]) -> list[tuple[object, ...]]:
    if not rows:
        return []

    metric_keys = list(rows[0]["_metric_keys"])
    param_keys = list(rows[0]["_param_keys"])
    header = ["rank", *metric_keys, "model", *ALWAYS_PARAMS, *param_keys, "duration_s", "run_id"]

    table_rows: list[tuple[object, ...]] = [tuple(header)]
    for rank, row in enumerate(rows, start=1):
        table_rows.append(
            tuple(
                [
                    rank,
                    *[_format_metric(row.get(key)) for key in metric_keys],
                    row.get("model", ""),
                    *[row.get(key, "") for key in ALWAYS_PARAMS],
                    *[row.get(key, "") for key in param_keys],
                    f"{row['duration_s']:.1f}",
                    row["run_id"],
                ]
            )
        )
    return table_rows


def infer_param_keys(runs: list[dict[str, Any]]) -> tuple[str, ...]:
    values_by_key: dict[str, set[str]] = {}
    for run in runs:
        config = run.get("config_json") or {}
        for key, value in (config.get("params") or {}).items():
            if key in ALWAYS_PARAMS:
                continue
            values_by_key.setdefault(key, set()).add(str(value))

    # Default comparison should focus on variables, not constants like a fixed flash_attn mode.
    return tuple(sorted(key for key, values in values_by_key.items() if len(values) > 1))


def _build_compare_row(
    run: dict[str, Any], *, metric_keys: tuple[str, ...], param_keys: tuple[str, ...]
) -> dict[str, Any]:
    config = run.get("config_json") or {}
    trial_params = config.get("params") or {}
    metrics = run.get("metrics") or {}
    prompt_tokens = config.get("prompt_tokens")
    gen_tokens = config.get("gen_tokens")

    row: dict[str, Any] = {
        "run_id": run["run_id"],
        "spec_hash": run.get("spec_hash", ""),
        "trial_name": run["trial_name"],
        "status": run["status"],
        "duration_s": run["duration_s"],
        "model": _model_name(config.get("model")),
        "prompt_tokens": prompt_tokens if prompt_tokens is not None else "",
        "gen_tokens": gen_tokens if gen_tokens is not None else "",
        "_metric_keys": metric_keys,
        "_param_keys": param_keys,
    }

    for key in param_keys:
        if key.startswith("system."):
            row[key] = _lookup_dotted(run.get("system_json") or {}, key.removeprefix("system."))
        else:
            row[key] = trial_params.get(key, config.get(key, ""))
    for key in metric_keys:
        row[key] = metric_value(key, metrics, prompt_tokens=prompt_tokens, gen_tokens=gen_tokens)
    return row


def _model_name(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or "")
    if value is None:
        return ""
    return str(value)


def _lookup_dotted(data: dict[str, Any], key: str) -> object:
    current: object = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return ""
        current = current[part]
    return current


def _dedupe_latest_configs(
    rows: list[dict[str, Any]], *, param_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    seen: set[tuple[object, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        signature = (
            ("spec_hash", row["spec_hash"])
            if row.get("spec_hash")
            else tuple(row.get(key, "") for key in (*IDENTITY_COLUMNS, *param_keys))
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(row)
    return deduped


def metric_value(
    key: str, metrics: dict[str, float], *, prompt_tokens: int | None, gen_tokens: int | None
) -> float | None:
    if key == "pp_tps":
        return prompt_tokens_per_second(metrics, prompt_tokens)
    if key == "tg_tps":
        return generation_tokens_per_second(metrics, gen_tokens)
    return metrics.get(key)


def prompt_tokens_per_second(metrics: dict[str, float], prompt_tokens: int | None) -> float | None:
    if prompt_tokens is not None:
        direct = metrics.get(f"pp{prompt_tokens}.tokens_per_second")
        if direct is not None:
            return direct

    for prefix in _metric_prefixes(metrics):
        n_prompt = _metric_int(metrics, prefix, "n_prompt")
        n_gen = _metric_int(metrics, prefix, "n_gen")
        prompt_matches = prompt_tokens is None or n_prompt == prompt_tokens
        if n_prompt and prompt_matches and not n_gen:
            return _tokens_per_second_for_prefix(metrics, prefix)
    return _first_metric_matching(metrics, r"^pp\d+\.tokens_per_second$")


def generation_tokens_per_second(metrics: dict[str, float], gen_tokens: int | None) -> float | None:
    if gen_tokens is not None:
        direct = metrics.get(f"tg{gen_tokens}.tokens_per_second")
        if direct is not None:
            return direct

    for prefix in _metric_prefixes(metrics):
        n_prompt = _metric_int(metrics, prefix, "n_prompt")
        n_gen = _metric_int(metrics, prefix, "n_gen")
        gen_matches = gen_tokens is None or n_gen == gen_tokens
        if n_gen and gen_matches and not n_prompt:
            return _tokens_per_second_for_prefix(metrics, prefix)
    return _first_metric_matching(metrics, r"^tg\d+\.tokens_per_second$")


def _tokens_per_second_for_prefix(metrics: dict[str, float], prefix: str) -> float | None:
    value = metrics.get(f"{prefix}.tokens_per_second")
    if value is not None:
        return value
    return metrics.get(f"{prefix}.avg_ts")


def _metric_prefixes(metrics: dict[str, float]) -> list[str]:
    prefixes = set()
    for key in metrics:
        if "." in key:
            prefixes.add(key.rsplit(".", 1)[0])
    return sorted(prefixes)


def _metric_int(metrics: dict[str, float], prefix: str, key: str) -> int:
    return int(metrics.get(f"{prefix}.{key}") or 0)


def _first_metric_matching(metrics: dict[str, float], pattern: str) -> float | None:
    compiled = re.compile(pattern)
    for key, value in metrics.items():
        if compiled.match(key):
            return value
    return None


def _sort_key(value: object, *, ascending: bool) -> tuple[int, float]:
    if not isinstance(value, int | float):
        return (1, 0.0)
    numeric_value = float(value)
    return (0, numeric_value if ascending else -numeric_value)


def _format_metric(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value:.3f}"
    return ""
