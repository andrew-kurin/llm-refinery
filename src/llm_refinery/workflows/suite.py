from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.parse import urlparse

from rich.console import Console

from llm_refinery.config import TuneConfig
from llm_refinery.http_load import load_http_load_config
from llm_refinery.lm_eval import LmEvalConfig, LmEvalTarget, run_lm_eval
from llm_refinery.utils.sanity import run_api_sanity_check
from llm_refinery.utils.system import get_system_snapshot, is_port_listening

console = Console()


class BenchmarkSuiteWorkflow:
    def __init__(
        self,
        config: TuneConfig,
        limit: int | None = 50,
        tasks: str = "ifeval,gsm8k",
        max_length: int = 8192,
        eos_string: str = "<turn|>",
        gen_kwargs: str | None = None,
        include_path: Path | None = None,
        run_lm_eval: bool = True,
        run_http_load: bool = False,
        require_clean: bool = True,
        llama_cpp_base_url: str = "http://127.0.0.1:8080/v1/chat/completions",
        http_load_config: Path | None = None,
        target_name: str | None = None,
        api_model: str = "local-model",
    ):
        self.config = config
        self.limit = limit
        self.tasks = tasks
        self.max_length = max_length
        self.eos_string = eos_string
        self.gen_kwargs = gen_kwargs
        self.include_path = include_path
        self.run_lm_eval = run_lm_eval
        self.run_http_load = run_http_load
        self.require_clean = require_clean
        self.llama_cpp_base_url = llama_cpp_base_url
        self.http_load_config = http_load_config
        self.target_name = target_name
        self.api_model = api_model

        self.log_dir = Path("results/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        console.print(f"[bold blue]==>[/bold blue] {message}")

    def _error(self, message: str) -> None:
        raise RuntimeError(message)

    def preflight(self) -> None:
        self._log("Performing preflight checks...")

        llama_port = _port_from_url(self.llama_cpp_base_url)
        if not is_port_listening(llama_port):
            self._error(
                f"no llama.cpp server listening on :{llama_port}; "
                "start it first, then rerun this command"
            )

        if self.require_clean:
            for port in [8081, 8082, 8083]:
                if port != llama_port and is_port_listening(port):
                    self._error(
                        f"port :{port} is listening; stop other MLX/model servers "
                        "for clean timing"
                    )

        self._log("Memory/process snapshot before")
        console.print(get_system_snapshot())

        self._log("Sanity check: reasoning off / content present")
        sanity = run_api_sanity_check(self.llama_cpp_base_url, model_name=self.api_model)
        if not sanity["success"]:
            self._error(str(sanity["error"]))

        console.print(f"    elapsed_s={sanity['elapsed_s']}")
        console.print(f"    content_len={sanity['content_len']}")
        console.print(f"    reasoning_len={sanity['reasoning_len']}")
        console.print(f"    finish_reason={sanity['finish_reason']}")
        console.print(f"    content_preview={sanity['content_preview']}")

    def run_quality(self) -> None:
        if not self.run_lm_eval:
            return

        self._log(f"Running lm-eval quality (tasks={self.tasks}, limit={self.limit})")
        try:
            run_lm_eval(
                LmEvalConfig(
                    target="llama_cpp",
                    limit=self.limit,
                    tasks=self.tasks,
                    max_length=self.max_length,
                    eos_string=self.eos_string,
                    gen_kwargs=self.gen_kwargs,
                    include_path=self.include_path,
                    suite_name=self.config.name,
                    database=self.config.database,
                    targets={
                        "llama_cpp": LmEvalTarget(
                            name="llama_cpp",
                            model=self.api_model,
                            base_url=self.llama_cpp_base_url,
                        )
                    },
                )
            )
        except RuntimeError as exc:
            self._error(str(exc))

    def run_load(self) -> None:
        if not self.run_http_load:
            return
        if not self.http_load_config:
            self._error("--http-load-config is required when HTTP load is enabled")

        http_config = load_http_load_config(self.http_load_config)
        self._log(f"Running HTTP load (target={self.target_name or 'all'})")

        cmd = ["uv", "run", "llm-refinery", "http-load", str(self.http_load_config)]
        if self.target_name:
            cmd.extend(["--target", self.target_name])

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            self._error(f"HTTP load failed: {exc}")

        self._log("HTTP comparison for this suite")
        compare_cmd = [
            "uv",
            "run",
            "llm-refinery",
            "compare",
            str(http_config.database),
            "--suite",
            http_config.name,
            "--metric",
            "latency_p95_s",
            "--metric",
            "ttft_p95_s",
            "--metric",
            "completion_tokens_per_second",
            "--metric",
            "check_pass_rate",
            "--sort",
            "latency_p95_s",
            "--ascending",
            "--param",
            "target",
            "--param",
            "provider",
            "--param",
            "scenario",
            "--param",
            "concurrency",
            "--limit",
            "20",
        ]

        try:
            subprocess.run(compare_cmd, check=True)
        except subprocess.CalledProcessError as exc:
            self._error(f"comparison failed: {exc}")

    def run_post_analysis(self) -> None:
        self._log("Memory/process snapshot after")
        console.print(get_system_snapshot())
        self._log("Done.")

    def execute(self) -> None:
        self.preflight()
        self.run_quality()
        self.run_load()
        self.run_post_analysis()


def _port_from_url(url: str) -> int:
    parsed = urlparse(url)
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    return 80
