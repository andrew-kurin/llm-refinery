from llama_tune.config import TuneConfig, expand_trials


def test_expand_trials_cartesian_product():
    config = TuneConfig.from_mapping(
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
    config = TuneConfig.from_mapping(
        {
            "name": "suite",
            "models": [{"name": "m", "hf": "repo:model"}],
            "sweep": {"parallel": [1, 2]},
            "bench": {"prompt_tokens": [128, 256], "gen_tokens": [32, 64]},
        }
    )

    trials = expand_trials(config, include_bench_dimensions=False)

    assert len(trials) == 2
    assert all(trial.prompt_tokens is None for trial in trials)
    assert all(trial.gen_tokens is None for trial in trials)


def test_eval_config_defaults_and_overrides():
    default_config = TuneConfig.from_mapping(
        {"name": "suite", "models": [{"name": "m", "hf": "repo:model"}]}
    )
    assert default_config.eval.tasks == "ifeval,gsm8k"
    assert default_config.eval.limit == 50
    assert default_config.eval.max_length == 8192
    assert default_config.eval.eos_string == "<turn|>"

    qwen_config = TuneConfig.from_mapping(
        {
            "name": "suite",
            "models": [{"name": "m", "hf": "repo:model"}],
            "eval": {
                "tasks": "ifeval",
                "limit": "all",
                "max_length": 4096,
                "eos_string": "<|im_end|>",
                "gen_kwargs": "enable_thinking=False",
            },
        }
    )
    assert qwen_config.eval.tasks == "ifeval"
    assert qwen_config.eval.limit is None
    assert qwen_config.eval.max_length == 4096
    assert qwen_config.eval.eos_string == "<|im_end|>"
    assert qwen_config.eval.gen_kwargs == "enable_thinking=False"
