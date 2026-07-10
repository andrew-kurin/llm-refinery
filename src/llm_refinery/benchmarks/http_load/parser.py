from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_refinery.benchmarks.http_load.metrics import summarize_request_results
from llm_refinery.benchmarks.http_load.models import RequestResult


def reparse_http_load_run(run: dict[str, Any]) -> dict[str, float]:
    artifact = (run.get("artifacts") or {}).get("responses")
    if not artifact:
        raise FileNotFoundError("HTTP load run has no responses artifact")
    path = Path(artifact["path"])
    results = [
        _request_result(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    params = (run.get("config_json") or {}).get("params") or {}
    concurrency = int(params.get("concurrency") or 1)
    max_tokens = int(params.get("max_tokens") or run["config_json"].get("gen_tokens") or 0)
    measurement = (run.get("artifacts") or {}).get("measurement")
    if measurement:
        measurement_data = json.loads(Path(measurement["path"]).read_text(encoding="utf-8"))
        wall_duration_s = float(measurement_data["wall_duration_s"])
    else:
        wall_duration_s = float(run.get("duration_s") or 0.0)
    return summarize_request_results(
        results,
        wall_duration_s=wall_duration_s,
        concurrency=concurrency,
        max_tokens=max_tokens,
    )


def _request_result(raw: dict[str, Any]) -> RequestResult:
    fields = RequestResult.__dataclass_fields__
    return RequestResult(**{key: raw[key] for key in fields if key in raw})
