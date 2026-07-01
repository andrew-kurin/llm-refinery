from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

TARGET_CHOICES = ("llama_cpp", "ollama", "mlx_e4b", "mlx_26b", "both", "all")
TARGET_ORDER = ("llama_cpp", "ollama", "mlx_e4b", "mlx_26b")


@dataclass(frozen=True)
class LmEvalTarget:
    name: str
    model: str
    base_url: str


@dataclass(frozen=True)
class LmEvalConfig:
    target: str = "llama_cpp"
    limit: int | None = 50
    tasks: str = "ifeval,gsm8k"
    num_concurrent: int = 1
    max_retries: int = 3
    max_length: int = 16384
    eos_string: str = "<turn|>"
    log_samples: bool = False
    gen_kwargs: str | None = None
    output_root: Path = Path("results/lm_eval")
    offline: bool = True
    model_backend: str = "local-chat-completions"
    apply_chat_template: bool = True
    include_path: Path | None = None
    suite_name: str = "lm-eval"
    database: Path = Path("results/llm_refinery.duckdb")
    targets: dict[str, LmEvalTarget] = field(default_factory=dict)


def default_targets(env: dict[str, str] | None = None) -> dict[str, LmEvalTarget]:
    env = os.environ if env is None else env
    return {
        "llama_cpp": LmEvalTarget(
            name="llama_cpp",
            model=env.get("LLAMA_CPP_MODEL", "local-model"),
            base_url=env.get(
                "LLAMA_CPP_BASE_URL", "http://127.0.0.1:8080/v1/chat/completions"
            ),
        ),
        "ollama": LmEvalTarget(
            name="ollama",
            model=env.get("OLLAMA_MODEL", "gemma4:26b"),
            base_url=env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1/chat/completions"),
        ),
        "mlx_e4b": LmEvalTarget(
            name="mlx_e4b",
            model=env.get("MLX_E4B_MODEL", "mlx-community/gemma-4-e4b-it-OptiQ-4bit"),
            base_url=env.get("MLX_E4B_BASE_URL", "http://127.0.0.1:8081/v1/chat/completions"),
        ),
        "mlx_26b": LmEvalTarget(
            name="mlx_26b",
            model=env.get("MLX_26B_MODEL", "mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit"),
            base_url=env.get("MLX_26B_BASE_URL", "http://127.0.0.1:8082/v1/chat/completions"),
        ),
    }


def resolve_target_names(target: str) -> list[str]:
    if target == "both":
        return ["llama_cpp", "ollama"]
    if target == "all":
        return list(TARGET_ORDER)
    if target not in TARGET_ORDER:
        raise ValueError(f"unknown lm-eval target: {target}")
    return [target]
