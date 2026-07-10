from __future__ import annotations

import os
from collections.abc import Mapping

from llm_refinery.core.endpoints import OPENAI_CHAT, Endpoint

TARGET_ORDER = (
    "llama_cpp",
    "ollama",
    "mlx_e4b",
    "mlx_26b",
    "ollama_31b",
    "ollama_31b_mlx",
    "cerebras_31b",
)


def default_targets(env: Mapping[str, str] | None = None) -> dict[str, Endpoint]:
    resolved_env = os.environ if env is None else env
    definitions = {
        "llama_cpp": (
            "LLAMA_CPP_MODEL",
            "local-model",
            "LLAMA_CPP_BASE_URL",
            "http://127.0.0.1:8080/v1",
            None,
        ),
        "ollama": (
            "OLLAMA_MODEL",
            "gemma4:26b",
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434/v1",
            None,
        ),
        "mlx_e4b": (
            "MLX_E4B_MODEL",
            "mlx-community/gemma-4-e4b-it-OptiQ-4bit",
            "MLX_E4B_BASE_URL",
            "http://127.0.0.1:8081/v1",
            None,
        ),
        "mlx_26b": (
            "MLX_26B_MODEL",
            "mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit",
            "MLX_26B_BASE_URL",
            "http://127.0.0.1:8082/v1",
            None,
        ),
        "ollama_31b": (
            "OLLAMA_31B_MODEL",
            "gemma4:31b",
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434/v1",
            None,
        ),
        "ollama_31b_mlx": (
            "OLLAMA_31B_MLX_MODEL",
            "gemma4:31b-mlx",
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434/v1",
            None,
        ),
        "cerebras_31b": (
            "CEREBRAS_31B_MODEL",
            "gemma-4-31b",
            "CEREBRAS_BASE_URL",
            "https://api.cerebras.ai/v1",
            "CEREBRAS_API_KEY",
        ),
    }
    return {
        name: Endpoint(
            name=name,
            protocol=OPENAI_CHAT,
            model=resolved_env.get(model_env, default_model),
            base_url=resolved_env.get(url_env, default_url),
            api_key_env=api_key_env,
        )
        for name, (
            model_env,
            default_model,
            url_env,
            default_url,
            api_key_env,
        ) in definitions.items()
    }
