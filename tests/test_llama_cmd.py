from llama_tune.config import TuneConfig, expand_trials
from llama_tune.llama_cmd import build_bench_command, build_server_command, shell_join


def test_build_bench_command_uses_unified_llama_cli_and_flags():
    config = TuneConfig.from_mapping(
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
    config = TuneConfig.from_mapping(
        {
            "name": "suite",
            "models": [{"name": "m", "path": "/models/model.gguf"}],
            "defaults": {"ctx_size": 4096},
            "server": {"extra_args": ["--host", "127.0.0.1"]},
        }
    )
    trial = expand_trials(config, include_bench_dimensions=False)[0]

    cmd = build_server_command(config, trial)

    assert shell_join(cmd) == "llama server -m /models/model.gguf --ctx-size 4096 --host 127.0.0.1"


def test_build_server_command_uses_llama_hf_file_short_flag():
    config = TuneConfig.from_mapping(
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
    trial = expand_trials(config, include_bench_dimensions=False)[0]

    cmd = build_server_command(config, trial)

    assert cmd[:5] == ["llama", "server", "-hf", "repo/model", "-hff"]
    assert cmd[5] == "model-Q4_K_M.gguf"
