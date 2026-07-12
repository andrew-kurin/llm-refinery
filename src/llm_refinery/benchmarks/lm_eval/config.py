from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from llm_refinery.benchmarks.lm_eval.presets import TARGET_ORDER
from llm_refinery.core.config import ConfigError
from llm_refinery.core.endpoints import Endpoint
from llm_refinery.core.http_safety import PinnedHttpRoute


@dataclass(frozen=True)
class LmEvalConfig:
    target: str = "llama_cpp"
    limit: int | None = 50
    tasks: str = "ifeval,gsm8k"
    num_concurrent: int = 1
    max_retries: int = 3
    max_length: int = 16384
    eos_string: str | None = None
    tokenizer: str | None = None
    metadata: str | None = None
    log_samples: bool = False
    num_fewshot: int | None = None
    gen_kwargs: str | None = None
    output_root: Path = Path("results/lm_eval")
    offline: bool = True
    model_backend: str = "local-chat-completions"
    package_spec: str = "lm-eval[api]==0.4.12"
    extra_packages: tuple[str, ...] = ()
    apply_chat_template: bool = True
    include_path: Path | None = None
    trust_env: bool = False
    ca_bundle: Path | None = None
    pinned_route: PinnedHttpRoute | None = None
    suite_name: str = "lm-eval"
    database: Path = Path("results/llm_refinery.duckdb")
    targets: dict[str, Endpoint] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target.strip():
            raise ConfigError("lm-eval target cannot be empty")
        if self.limit is not None and self.limit <= 0:
            raise ConfigError("lm-eval limit must be positive or None")
        if self.num_concurrent <= 0:
            raise ConfigError("lm-eval num_concurrent must be positive")
        if self.max_retries < 0:
            raise ConfigError("lm-eval max_retries cannot be negative")
        if self.max_length <= 0:
            raise ConfigError("lm-eval max_length must be positive")
        if not isinstance(self.trust_env, bool):
            raise ConfigError("lm-eval trust_env must be a boolean")
        if self.trust_env and self.num_concurrent > 1 and self.pinned_route is None:
            raise ConfigError(
                "lm-eval trust_env is supported only with num_concurrent=1 because the "
                "pinned asynchronous API client does not honor proxy environment variables"
            )
        if self.ca_bundle is not None and not self.ca_bundle.is_file():
            raise ConfigError(f"lm-eval ca_bundle is not a file: {self.ca_bundle}")
        if not self.package_spec.strip():
            raise ConfigError("lm-eval package_spec cannot be empty")
        if any(not package.strip() for package in self.extra_packages):
            raise ConfigError("lm-eval extra package specs cannot be empty")
        if self.tokenizer and self.model_backend == "local-chat-completions":
            raise ConfigError(
                "lm-eval tokenizer is not supported by the local-chat-completions "
                "backend: it ignores client-side tokenization and token-aware truncation"
            )
        if self.metadata is not None:
            try:
                metadata = json.loads(self.metadata)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"lm-eval metadata must be valid JSON: {exc}") from exc
            if not isinstance(metadata, dict):
                raise ConfigError("lm-eval metadata must be a JSON object")


def resolve_target_names(target: str, available: set[str] | None = None) -> list[str]:
    available = set(TARGET_ORDER) if available is None else available
    if target == "both":
        selected = ["llama_cpp", "ollama"]
    elif target == "all":
        selected = [name for name in TARGET_ORDER if name in available]
    else:
        selected = [target]
    missing = set(selected) - available
    if missing:
        raise ValueError(f"unknown lm-eval target(s): {', '.join(sorted(missing))}")
    return selected
