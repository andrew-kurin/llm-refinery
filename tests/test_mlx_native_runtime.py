import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "mlx_native_runtime", Path(__file__).resolve().parents[1] / "tools" / "mlx_native_runtime.py"
)
assert spec is not None and spec.loader is not None
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def test_native_observer_preserves_mutable_arguments_results_and_return_value():
    data = {}
    report = SimpleNamespace(data=data, update=lambda **values: data.update(values))
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[1, 106, 50]))
    tokenizer = SimpleNamespace(stopping_criteria=SimpleNamespace(eos_token_ids=[1, 106, 50]))
    values = {"prefill_step_size": 512, "seed": 42, "max_tokens": 32}
    result = object()

    def generate(model, tokenizer, kwargs):
        assert kwargs is values
        assert kwargs.pop("seed") == 42
        yield result
        return "finished"

    iterator = native.observe_native_diffusion(report, generate)(model, tokenizer, values)
    assert next(iterator) is result
    with pytest.raises(StopIteration) as stopped:
        next(iterator)
    assert stopped.value.value == "finished"
    assert values == {"prefill_step_size": 512, "max_tokens": 32}
    assert data["observed_diffusion_arguments"]["seed"] == 42
    assert data["observed_diffusion_arguments"]["prefill_step_size"] == 512
    assert data["requests_started"] == data["requests_completed"] == 1
    assert data["config_stop_token_count"] == data["stop_token_count"] == 3


def test_native_observer_preserves_generation_errors():
    data = {}
    report = SimpleNamespace(data=data, update=lambda **values: data.update(values))
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=1))
    tokenizer = SimpleNamespace(stopping_criteria=SimpleNamespace(eos_token_ids=[1]))

    def generate(model, tokenizer, kwargs):
        yield "first"
        raise RuntimeError("model failure")

    iterator = native.observe_native_diffusion(report, generate)(model, tokenizer, {})
    assert next(iterator) == "first"
    with pytest.raises(RuntimeError, match="model failure"):
        next(iterator)
    assert data["requests_completed"] == 1
    assert data["config_stop_token_count"] == 1
