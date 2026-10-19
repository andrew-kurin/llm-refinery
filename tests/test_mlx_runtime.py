import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "mlx_runtime", Path(__file__).resolve().parents[1] / "tools" / "mlx_runtime.py"
)
assert spec is not None and spec.loader is not None
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
ANSWER, RuntimeReport, recall_prompt = runtime.ANSWER, runtime.RuntimeReport, runtime.recall_prompt


def test_runtime_observers_preserve_cache_and_diffusion_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "version", lambda _name: "pinned")
    path = tmp_path / "runtime.json"
    report = RuntimeReport(path)
    caches = [SimpleNamespace(keep=0, max_size=1024), SimpleNamespace()]

    class Model:
        def make_cache(self, max_size=None):
            assert max_size is None
            return caches

    model = Model()
    report.observe_cache(model)
    assert model.make_cache() is caches

    def diffuse(model, *, prefill_step_size=None):
        return model, prefill_step_size

    observed = report.observe_diffusion(diffuse)
    assert observed(model) == (model, None)
    data = json.loads(path.read_text())
    assert data["observed_cache_layout"][0]["keep"] == 0
    assert data["observed_cache_layout"][0]["max_size"] == 1024
    assert data["observed_diffusion_arguments"]["prefill_step_size"] is None
    assert data["diffusion_supported_arguments"] == ["prefill_step_size"]
    instance = data["instance_id"]
    assert observed(model, prefill_step_size=512) == (model, 512)
    updated = json.loads(path.read_text())
    assert updated["observed_diffusion_arguments"]["prefill_step_size"] == 512
    assert updated["instance_id"] == instance


def test_recall_prompt_has_one_needle_at_the_requested_length():
    # A whitespace tokenizer exercises the fitting logic without downloading a model.
    def count(prompt):
        return len(prompt.split())

    near_end = recall_prompt(count, 8193)
    early = recall_prompt(count, 8193, "start")
    assert count(near_end) == count(early) == 8193
    assert near_end.count(ANSWER) == early.count(ANSWER) == 1
    assert near_end.index(ANSWER) > len(near_end) * 0.9
    assert early.index(ANSWER) < 100


def test_diffusion_forwards_server_prefill_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "version", lambda _name: "pinned")
    report = RuntimeReport(tmp_path / "runtime.json")
    server = SimpleNamespace(step=4096)

    def diffuse(model, prefill_step_size=None):
        return model, prefill_step_size

    wrapped = report.observe_diffusion(diffuse, prefill_step_size_getter=lambda: server.step)
    assert wrapped("model") == ("model", 4096)
    assert report.data["observed_diffusion_arguments"]["prefill_step_size"] == 4096
    server.step = 512
    assert wrapped("model", prefill_step_size=None) == ("model", 512)
    assert wrapped("model", 2048) == ("model", 2048)
    assert report.data["observed_diffusion_arguments"]["prefill_step_size"] == 2048


def test_diffusion_without_prefill_support_preserves_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "version", lambda _name: "pinned")
    report = RuntimeReport(tmp_path / "runtime.json")

    def diffuse(model, *, temperature):
        return model, temperature

    wrapped = report.observe_diffusion(diffuse, prefill_step_size_getter=lambda: 512)
    assert wrapped("model", temperature=0) == ("model", 0)
    assert report.data["diffusion_supported_arguments"] == []


def test_diffusion_isolates_aliased_stop_tokens_across_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "version", lambda _name: "pinned")
    report = RuntimeReport(tmp_path / "runtime.json")
    config_ids = [1, 106, 50]
    criteria = SimpleNamespace(eos_token_ids=config_ids)
    tokenizer = SimpleNamespace(stopping_criteria=criteria)

    def diffuse(tokenizer):
        # mlx-vlm 0.6.3 extends the criteria list with the config's EOS list.
        tokenizer.stopping_criteria.eos_token_ids.extend(config_ids)
        return set(tokenizer.stopping_criteria.eos_token_ids)

    wrapped = report.observe_diffusion(diffuse)
    for _ in range(40):
        # The native server resets the list from model.config before each call.
        criteria.eos_token_ids = config_ids
        assert wrapped(tokenizer) == {1, 106, 50}
        assert criteria.eos_token_ids == [1, 106, 50, 1, 106, 50]
        assert config_ids == [1, 106, 50]
