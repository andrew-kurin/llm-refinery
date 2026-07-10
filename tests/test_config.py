import pytest

from llm_refinery.benchmarks.llama_bench.config import LlamaSweepConfig, expand_trials
from llm_refinery.core.config import ConfigError


def test_llama_config_rejects_unknown_fields():
    with pytest.raises(ConfigError, match="unknown field.*http_load_config"):
        LlamaSweepConfig.from_mapping(
            {
                "models": [{"name": "m", "hf": "repo:model"}],
                "http_load_config": "ignored.yaml",
            }
        )


def test_expand_trials_cartesian_product():
    config = LlamaSweepConfig.from_mapping(
        {
            "name": "suite",
            "models": [{"name": "m", "hf": "repo:model"}],
            "defaults": {"ctx_size": 1024, "cache_type_k": "q4_0"},
            "sweep": {"cache_type_k": ["q4_0", "q8_0"], "parallel": [1, 2]},
            "bench": {"prompt_tokens": [128], "gen_tokens": [32, 64]},
        }
    )

    trials = expand_trials(config)

    assert len(trials) == 8
    assert trials[0].params["ctx_size"] == 1024
    assert {trial.params["cache_type_k"] for trial in trials} == {"q4_0", "q8_0"}
    assert {trial.params["parallel"] for trial in trials} == {1, 2}
    assert {trial.gen_tokens for trial in trials} == {32, 64}


def test_expand_server_trials_omit_bench_dimensions():
    config = LlamaSweepConfig.from_mapping(
        {
            "name": "suite",
            "models": [{"name": "m", "hf": "repo:model"}],
            "sweep": {"parallel": [1, 2]},
            "bench": {"prompt_tokens": [128, 256], "gen_tokens": [32, 64]},
        }
    )

    trials = expand_trials(config, kind="server")

    assert len(trials) == 2
    assert all(trial.prompt_tokens is None for trial in trials)
    assert all(trial.gen_tokens is None for trial in trials)


def test_trial_identity_includes_effective_benchmark_configuration():
    base = {
        "name": "suite",
        "models": [{"name": "m", "hf": "repo:model"}],
        "bench": {
            "prompt_tokens": [128],
            "gen_tokens": [32],
            "repetitions": 1,
            "params": {"threads": 1},
        },
    }
    first = expand_trials(LlamaSweepConfig.from_mapping(base))[0]
    base["bench"] = {
        "prompt_tokens": [128],
        "gen_tokens": [32],
        "repetitions": 9,
        "params": {"threads": 8},
        "extra_args": ["--verbose"],
    }
    second = expand_trials(LlamaSweepConfig.from_mapping(base))[0]

    assert first.key != second.key
    assert first.as_jsonable() != second.as_jsonable()
