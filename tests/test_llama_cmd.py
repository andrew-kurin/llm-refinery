from llm_refinery.benchmarks.llama_bench.command_builder import (
    build_bench_command,
    build_server_command,
    shell_join,
)
from llm_refinery.benchmarks.llama_bench.config import LlamaSweepConfig, expand_trials


def test_build_bench_command_uses_unified_llama_cli_and_flags():
    config = LlamaSweepConfig.from_mapping(
        {
            "name": "suite",
            "commands": {"bench": ["llama", "bench"], "server": ["llama", "server"]},
            "models": [{"name": "m", "hf": "repo:model"}],
            "defaults": {
                "ctx_size": 16384,
                "n_gpu_layers": "all",
                "mlock": True,
                "perf": True,
                "no_mmap": False,
            },
            "bench": {"prompt_tokens": [512], "gen_tokens": [128], "repetitions": 3},
        }
    )
    trial = expand_trials(config)[0]

    cmd = build_bench_command(config, trial)

    assert cmd == [
        "llama",
        "bench",
        "-hf",
        "repo:model",
        "--ctx-size",
        "16384",
        "--n-gpu-layers",
        "all",
        "--mlock",
        "--perf",
        "-p",
        "512",
        "-n",
        "128",
        "-r",
        "3",
        "-o",
        "json",
    ]
    assert "--no-mmap" not in cmd


def test_build_server_command_omits_bench_dimensions():
    config = LlamaSweepConfig.from_mapping(
        {
            "name": "suite",
            "models": [{"name": "m", "path": "/models/model.gguf"}],
            "defaults": {"ctx_size": 4096},
            "server": {"extra_args": ["--host", "127.0.0.1"]},
        }
    )
    trial = expand_trials(config, kind="server")[0]

    cmd = build_server_command(config, trial)

    assert shell_join(cmd) == "llama server -m /models/model.gguf --ctx-size 4096 --host 127.0.0.1"


def test_build_server_command_resolves_mtp_head(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = LlamaSweepConfig.from_mapping(
        {
            "name": "suite",
            "models": [{"name": "m", "hf": "repo/model"}],
            "server": {
                "params": {
                    "spec_type": "draft-mtp",
                    "mtp_head": {
                        "hf": "org/draft-repo",
                        "file": "MTP/head-Q8_0.gguf",
                    },
                }
            },
        }
    )
    trial = expand_trials(config, kind="server")[0]

    cmd = build_server_command(config, trial)

    assert "--model-draft" in cmd
    draft_path = cmd[cmd.index("--model-draft") + 1]
    assert draft_path == str(tmp_path / ".local/share/llm-refinery/mtp/head-Q8_0.gguf")
    assert "--spec-type" in cmd
    assert "draft-mtp" in cmd


def test_build_server_command_uses_custom_model_flag_for_non_llama_servers():
    config = LlamaSweepConfig.from_mapping(
        {
            "name": "suite",
            "commands": {"server": ["uvx", "--from", "mlx-vlm", "mlx_vlm.server"]},
            "models": [{"name": "m", "hf": "mlx-community/model-4bit"}],
            "server": {
                "model_flag": "--model",
                "params": {"host": "127.0.0.1", "port": 8082},
            },
        }
    )
    trial = expand_trials(config, kind="server")[0]

    cmd = build_server_command(config, trial)

    assert cmd == [
        "uvx",
        "--from",
        "mlx-vlm",
        "mlx_vlm.server",
        "--model",
        "mlx-community/model-4bit",
        "--host",
        "127.0.0.1",
        "--port",
        "8082",
    ]


def test_build_server_command_uses_llama_hf_file_short_flag():
    config = LlamaSweepConfig.from_mapping(
        {
            "name": "suite",
            "models": [
                {
                    "name": "m",
                    "hf": "repo/model",
                    "params": {"hff": "model-Q4_K_M.gguf"},
                }
            ],
        }
    )
    trial = expand_trials(config, kind="server")[0]

    cmd = build_server_command(config, trial)

    assert cmd[:5] == ["llama", "server", "-hf", "repo/model", "-hff"]
    assert cmd[5] == "model-Q4_K_M.gguf"
