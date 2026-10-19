"""Exercise the pinned runtime's real factory and forwarding code without weights."""

import argparse
import json
import os
from collections import Counter
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from huggingface_hub import hf_hub_download
from mlx_runtime import isolate_stop_token_ids
from mlx_vlm.models.diffusion_gemma.language import EncoderModel
from mlx_vlm.server import generation
from mlx_vlm.utils import StoppingCriteria, load_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("results/mlx-diffusion-probe.json"))
args_cli = parser.parse_args()
revision = "0d2cee4aa112548df2fad35a7f017eeb1ac8f352"
config_path = hf_hub_download(
    "mlx-community/diffusiongemma-26B-A4B-it-4bit", "config.json", revision=revision
)
config = json.loads(Path(config_path).read_text())
text_config = config["text_config"]
stand_in = SimpleNamespace(
    text_config=SimpleNamespace(
        layer_types=text_config["layer_types"], sliding_window=text_config["sliding_window"]
    )
)
caches = EncoderModel.make_cache(stand_in)
layout = Counter(
    (type(cache).__name__, getattr(cache, "keep", None), getattr(cache, "max_size", None))
    for cache in caches
)
trace = {
    "method": "Actual factory method with checkpoint configuration; no model weights loaded",
    "checkpoint_revision": revision,
    "packages": {name: version(name) for name in ["mlx", "mlx-lm", "mlx-vlm"]},
    "cache_layout": [
        {"type": typ, "keep": keep, "max_size": size, "layers": count}
        for (typ, keep, size), count in layout.items()
    ],
}
assert sum(type(c).__name__ == "RotatingKVCache" and c.keep == 0 for c in caches) == 25
assert sum(type(c).__name__ == "KVCache" for c in caches) == 5

# Use a stub generator to capture arguments from the actual server method.
# The context manager is replaced only to avoid doing unrelated Metal wiring.
os.environ["PREFILL_STEP_SIZE"] = "4096"
os.environ["MAX_KV_SIZE"] = "8192"
original_stream = generation.stream_diffusion_generate
original_wired = generation.wired_limit


def capture(*args, **kwargs):
    trace["forwarding"] = {
        "configured_prefill_step_size": generation.get_prefill_step_size(),
        "configured_max_kv_size": os.environ["MAX_KV_SIZE"],
        "received_keyword_names": list(kwargs),
        "prefill_step_size_forwarded": "prefill_step_size" in kwargs,
        "max_kv_size_forwarded": "max_kv_size" in kwargs,
    }
    yield from ()


try:
    generation.stream_diffusion_generate = capture
    generation.wired_limit = lambda *args, **kwargs: nullcontext()
    server = SimpleNamespace(
        model=object(), processor=object(), tokenizer=SimpleNamespace(all_special_ids=[])
    )
    args = SimpleNamespace(logits_processors=None, seed=None, max_tokens=1, temperature=0)
    generation.ResponseGenerator._generate_diffusion(
        server, 1, None, {"input_ids": mx.array([[1]])}, args, set()
    )
finally:
    generation.stream_diffusion_generate = original_stream
    generation.wired_limit = original_wired

assert trace["forwarding"]["configured_prefill_step_size"] == 4096
assert not trace["forwarding"]["prefill_step_size_forwarded"]
assert not trace["forwarding"]["max_kv_size_forwarded"]
checks = []
for limit in [8192, 0]:
    os.environ["MAX_KV_SIZE"] = str(limit)
    for prompt_tokens in [4096, 8191, 8192, 8193, 16384]:
        accepted = True
        try:
            generation._check_configured_context_budget(prompt_tokens, 32)
        except generation.PromptTooLongError:
            accepted = False
        assert accepted == ((not limit) or prompt_tokens + 32 <= limit)
        checks.append(
            {
                "configured_limit": limit,
                "prompt_tokens": prompt_tokens,
                "max_tokens": 32,
                "accepted": accepted,
            }
        )
trace["context_admission_checks"] = checks

# load_config aliases the generation EOS list into the top-level configuration.
# The native server reset followed by stream_diffusion_generate's append doubles
# that same list on every request. Keep this reproduction small and bounded.
hf_hub_download(
    "mlx-community/diffusiongemma-26B-A4B-it-4bit", "generation_config.json", revision=revision
)
merged = load_config(Path(config_path).parent)
criteria = StoppingCriteria(merged["eos_token_id"], tokenizer=object())
stock_lengths = []
for _ in range(12):
    criteria.reset(merged["eos_token_id"])
    criteria.add_eos_token_ids(merged["generation_config"]["eos_token_id"])
    stock_lengths.append(len(criteria.eos_token_ids))
assert stock_lengths == [3 * 2**index for index in range(1, 13)]
merged = load_config(Path(config_path).parent)
criteria = StoppingCriteria(merged["eos_token_id"], tokenizer=object())
tokenizer = SimpleNamespace(stopping_criteria=criteria)
isolated_lengths = []
for _ in range(40):
    criteria.reset(merged["eos_token_id"])
    isolate_stop_token_ids(tokenizer)
    criteria.add_eos_token_ids(merged["generation_config"]["eos_token_id"])
    isolated_lengths.append(len(criteria.eos_token_ids))
assert isolated_lengths == [6] * 40
assert merged["eos_token_id"] == [1, 106, 50]
trace["stop_token_aliasing"] = {
    "stock_lengths_after_each_request": stock_lengths,
    "isolated_lengths_after_each_request": isolated_lengths,
    "isolated_config_ids": merged["eos_token_id"],
}
args_cli.output.parent.mkdir(parents=True, exist_ok=True)
args_cli.output.write_text(json.dumps(trace, indent=2) + "\n")
print(json.dumps(trace, indent=2))
