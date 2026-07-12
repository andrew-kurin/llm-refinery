from __future__ import annotations

import re
from collections.abc import Mapping

from llm_refinery.benchmarks.lm_eval.config import LmEvalConfig
from llm_refinery.core.config import ConfigError
from llm_refinery.core.endpoints import Endpoint
from llm_refinery.providers.openai_chat import validate_http_headers

_BEARER_AUTHORIZATION = re.compile(r"Bearer[ \t]+([^\s]+)", re.IGNORECASE)
_PROXY_ENVIRONMENT_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)
_CA_ENVIRONMENT_VARIABLES = (
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)


def build_lm_eval_command(config: LmEvalConfig, target: Endpoint) -> list[str]:
    validate_lm_eval_headers(target)
    output_path = str(config.output_root / target.name)
    base_url = (
        target.completions_url
        if config.model_backend == "local-completions"
        else target.chat_completions_url
    )
    model_args_parts = [
        f"model={target.model}",
        f"base_url={base_url}",
        f"num_concurrent={config.num_concurrent}",
        f"max_retries={config.max_retries}",
        f"max_length={config.max_length}",
    ]
    if config.eos_string:
        model_args_parts.append(f"eos_string={config.eos_string}")
    if config.tokenizer:
        model_args_parts.append(f"tokenizer={config.tokenizer}")
    model_args = ",".join(model_args_parts)

    cmd = [
        "uvx",
        "--from",
        config.package_spec,
        "--with",
        "langdetect",
        "--with",
        "immutabledict",
    ]
    for package in config.extra_packages:
        cmd.extend(["--with", package])
    child_command: list[str] = []
    strip_child_proxy = not config.trust_env or config.pinned_route is not None
    if strip_child_proxy or config.ca_bundle is not None:
        child_command.append("env")
        if strip_child_proxy:
            for name in _PROXY_ENVIRONMENT_VARIABLES:
                child_command.extend(["-u", name])
        if not config.trust_env or config.ca_bundle is not None:
            for name in _CA_ENVIRONMENT_VARIABLES:
                child_command.extend(["-u", name])
        if config.ca_bundle is not None:
            ca_bundle = str(config.ca_bundle)
            child_command.extend(
                [
                    f"SSL_CERT_FILE={ca_bundle}",
                    f"REQUESTS_CA_BUNDLE={ca_bundle}",
                    f"CURL_CA_BUNDLE={ca_bundle}",
                ]
            )
    child_command.extend(
        [
            "lm_eval",
            "--model",
            config.model_backend,
            "--model_args",
            model_args,
            "--tasks",
            config.tasks,
            "--batch_size",
            "1",
        ]
    )
    cmd.extend(child_command)

    if config.limit is not None:
        cmd.extend(["--limit", str(config.limit)])

    if config.num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(config.num_fewshot)])

    if config.apply_chat_template:
        cmd.append("--apply_chat_template")

    if config.include_path is not None:
        cmd.extend(["--include_path", str(config.include_path)])
    if config.log_samples:
        cmd.append("--log_samples")
    if config.gen_kwargs:
        cmd.extend(["--gen_kwargs", config.gen_kwargs])
    if config.metadata:
        cmd.extend(["--metadata", config.metadata])

    cmd.extend(["--output_path", output_path])
    return cmd


def validate_lm_eval_headers(target: Endpoint) -> None:
    """Reject headers that lm-eval cannot receive without exposing values in argv."""
    validate_http_headers(target.headers)
    authorization = [
        value for key, value in target.headers.items() if key.casefold() == "authorization"
    ]
    unsupported = sorted(key for key in target.headers if key.casefold() != "authorization")
    if unsupported:
        raise ConfigError(
            "lm-eval does not safely support custom endpoint headers; unsupported header "
            f"name(s): {', '.join(unsupported)}. Use api_key_env for Bearer authentication"
        )
    if len(authorization) > 1:
        raise ConfigError("lm-eval endpoint defines Authorization more than once")
    if authorization and _BEARER_AUTHORIZATION.fullmatch(authorization[0].strip()) is None:
        raise ConfigError(
            "lm-eval supports only a Bearer Authorization endpoint header; "
            "prefer api_key_env so the credential stays out of configuration and argv"
        )
    if authorization:
        validate_http_headers({"Authorization": authorization[0]})


def lm_eval_api_key(
    target: Endpoint,
    *,
    environ: Mapping[str, str],
) -> str | None:
    """Resolve target Bearer auth for the subprocess environment, never its argv."""
    validate_lm_eval_headers(target)
    for key, value in target.headers.items():
        if key.casefold() == "authorization":
            match = _BEARER_AUTHORIZATION.fullmatch(value.strip())
            assert match is not None  # validated above
            return match.group(1)
    if target.api_key_env:
        token = environ.get(target.api_key_env)
        if not token:
            raise ConfigError(
                f"lm-eval endpoint API key environment variable is not set: {target.api_key_env}"
            )
        validate_http_headers({"Authorization": f"Bearer {token}"})
        return token
    return None
