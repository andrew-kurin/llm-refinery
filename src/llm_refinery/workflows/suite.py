from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from rich.console import Console
from rich.table import Table

from llm_refinery.application.run_session import RunSession
from llm_refinery.benchmarks.http_load.config import load_http_load_config
from llm_refinery.benchmarks.http_load.runner import run_http_load
from llm_refinery.benchmarks.lm_eval.config import LmEvalConfig
from llm_refinery.benchmarks.lm_eval.runner import run_lm_eval
from llm_refinery.compare import build_compare_rows, build_compare_table_rows
from llm_refinery.core.runs import CompletedRun, RunSpec
from llm_refinery.storage.duckdb import ResultStore
from llm_refinery.utils.sanity import run_api_sanity_check
from llm_refinery.utils.system import get_system_snapshot, is_port_listening
from llm_refinery.workflows.suite_config import SuiteConfig

LmEvalRunner = Callable[..., list[CompletedRun]]
HttpLoadRunner = Callable[..., list[CompletedRun]]


@dataclass(frozen=True)
class SuiteResult:
    run: CompletedRun
    children: tuple[CompletedRun, ...]


class BenchmarkSuiteWorkflow:
    def __init__(
        self,
        config: SuiteConfig,
        *,
        console: Console | None = None,
        lm_eval_runner: LmEvalRunner = run_lm_eval,
        http_load_runner: HttpLoadRunner = run_http_load,
        port_listener: Callable[[int], bool] = is_port_listening,
        sanity_checker: Callable[..., dict[str, Any]] = run_api_sanity_check,
        system_snapshot: Callable[[], str] = get_system_snapshot,
    ) -> None:
        self.config = config
        self.console = console or Console()
        self.lm_eval_runner = lm_eval_runner
        self.http_load_runner = http_load_runner
        self.port_listener = port_listener
        self.sanity_checker = sanity_checker
        self.system_snapshot = system_snapshot

    def execute(self) -> SuiteResult:
        command = f"llm-refinery suite {self.config.name}"
        spec = RunSpec.create(
            benchmark_kind="suite",
            suite=self.config.name,
            label=self.config.name,
            command=command,
            config_json=self.config.safe_json(),
            database=self.config.database,
        )
        children: list[CompletedRun] = []
        with ResultStore(self.config.database) as store, RunSession(store, spec) as run:
            before_path = run.artifact("system_before", "system-before.txt", "text/plain")
            after_path = run.artifact("system_after", "system-after.txt", "text/plain")
            preflight_path = run.artifact("preflight", "preflight.json", "application/json")
            try:
                before_snapshot = self.system_snapshot()
                before_path.write_text(before_snapshot, encoding="utf-8")
                preflight_result = self.preflight(before_snapshot)
                preflight_path.write_text(
                    json.dumps(preflight_result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                children.extend(self.run_quality(store, parent_run_id=run.run_id))
                children.extend(self.run_load(store, parent_run_id=run.run_id))
            finally:
                after_snapshot = self.system_snapshot()
                after_path.write_text(after_snapshot, encoding="utf-8")
                self._log("Memory/process snapshot after")
                self.console.print(after_snapshot)

            failed_children = sum(child.status != "ok" for child in children)
            outcome = run.complete(
                status="ok" if failed_children == 0 else "failed",
                metrics={
                    "child_run_count": float(len(children)),
                    "failed_child_count": float(failed_children),
                },
                error=None if failed_children == 0 else f"{failed_children} child run(s) failed",
            )
            if failed_children:
                raise RuntimeError(f"{failed_children} suite child run(s) failed")
        self._log("Done.")
        return SuiteResult(run=outcome, children=tuple(children))

    def preflight(self, snapshot: str | None = None) -> dict[str, Any]:
        config = self.config.preflight
        if not config.enabled:
            return {"enabled": False}
        self._log("Performing preflight checks...")
        endpoint = self.config.endpoint
        parsed = urlparse(endpoint.base_url)
        endpoint_port = _port_from_url(endpoint.base_url)
        if parsed.hostname in {"127.0.0.1", "localhost", "::1"} and not self.port_listener(
            endpoint_port
        ):
            raise RuntimeError(
                f"no model server listening on :{endpoint_port}; start it first, then rerun"
            )

        if config.require_clean:
            for port in config.forbidden_ports:
                if port != endpoint_port and self.port_listener(port):
                    raise RuntimeError(
                        f"port :{port} is listening; stop other model servers for clean timing"
                    )

        self._log("Memory/process snapshot before")
        self.console.print(snapshot if snapshot is not None else self.system_snapshot())
        if not config.sanity_check:
            return {"enabled": True, "sanity_check": False}

        self._log("Sanity check: content present")
        sanity = self.sanity_checker(endpoint)
        if not sanity["success"]:
            raise RuntimeError(str(sanity["error"]))
        if (
            config.expected_response_model is not None
            and sanity.get("response_model") != config.expected_response_model
        ):
            raise RuntimeError(
                "endpoint returned model "
                f"{sanity.get('response_model')!r}; expected "
                f"{config.expected_response_model!r}"
            )
        for key in (
            "elapsed_s",
            "content_len",
            "reasoning_len",
            "finish_reason",
            "requested_model",
            "response_model",
            "model_matches",
            "content_preview",
        ):
            self.console.print(f"    {key}={sanity.get(key)}")
        return {"enabled": True, "sanity_check": True, "sanity": sanity}

    def run_quality(self, store: ResultStore, *, parent_run_id: str) -> list[CompletedRun]:
        quality = self.config.quality
        if not quality.enabled:
            return []
        endpoint = self.config.endpoint
        self._log(f"Running lm-eval quality (tasks={quality.tasks}, limit={quality.limit})")
        lm_config = LmEvalConfig(
            target=endpoint.name,
            limit=quality.limit,
            tasks=quality.tasks,
            max_length=quality.max_length,
            eos_string=quality.eos_string,
            tokenizer=quality.tokenizer,
            metadata=quality.metadata,
            num_fewshot=quality.num_fewshot,
            gen_kwargs=quality.gen_kwargs,
            include_path=quality.include_path,
            output_root=quality.output_root,
            package_spec=quality.package_spec,
            extra_packages=quality.extra_packages,
            offline=quality.offline,
            suite_name=self.config.name,
            database=self.config.database,
            log_samples=True,
            targets={endpoint.name: endpoint},
        )
        return self.lm_eval_runner(
            lm_config,
            parent_run_id=parent_run_id,
            store=store,
        )

    def run_load(self, store: ResultStore, *, parent_run_id: str) -> list[CompletedRun]:
        step = self.config.http_load
        if not step.enabled:
            return []
        assert step.config is not None
        http_config = load_http_load_config(step.config)
        targets = step.targets or (self.config.endpoint.name,)
        self._log(f"Running HTTP load (targets={','.join(targets)})")
        outcomes = self.http_load_runner(
            http_config,
            target_names=targets,
            scenario_names=step.scenarios,
            database_override=self.config.database,
            parent_run_id=parent_run_id,
            store=store,
        )
        self._print_http_comparison(store, http_config.name)
        return outcomes

    def _print_http_comparison(self, store: ResultStore, suite_name: str) -> None:
        runs = [run for run in store.comparison_runs() if run["suite"] == suite_name]
        rows = build_compare_rows(
            runs,
            metrics=(
                "observed_latency_p95_s",
                "visible_ttft_p95_s",
                "reasoning_ttft_p95_s",
                "tpot_p95_s",
                "completion_tokens_per_second",
                "check_pass_rate",
            ),
            params=("target", "protocol", "scenario", "concurrency"),
            sort_key="observed_latency_p95_s",
            ascending=True,
            limit=20,
        )
        table_rows = build_compare_table_rows(rows)
        if not table_rows:
            return
        self._log("HTTP comparison for this suite")
        table = Table(*[str(header) for header in table_rows[0]])
        for row in table_rows[1:]:
            table.add_row(*[str(cell) for cell in row])
        self.console.print(table)

    def _log(self, message: str) -> None:
        self.console.print(f"[bold blue]==>[/bold blue] {message}")


def _port_from_url(url: str) -> int:
    parsed = urlparse(url)
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    return 80
