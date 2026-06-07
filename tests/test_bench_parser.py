from llm_refinery.bench_parser import parse_llama_bench_metrics


def test_parse_json_bench_metrics():
    stdout = '[{"test":"tg128","t/s":42.5,"avg_ns":1234},{"test":"pp512","t/s":84.0}]'

    metrics = parse_llama_bench_metrics(stdout)

    assert metrics["tg128.tokens_per_second"] == 42.5
    assert metrics["tg128.avg_ns"] == 1234.0
    assert metrics["pp512.tokens_per_second"] == 84.0


def test_parse_llama_bench_json_metrics():
    stdout = """
[
  {"n_prompt":512,"n_gen":0,"avg_ts":303.7,"stddev_ts":11.0},
  {"n_prompt":0,"n_gen":128,"avg_ts":28.6,"samples_ts":[28.1,29.0]}
]
"""

    metrics = parse_llama_bench_metrics(stdout)

    assert metrics["pp512.tokens_per_second"] == 303.7
    assert metrics["pp512.tokens_per_second_stddev"] == 11.0
    assert metrics["tg128.tokens_per_second"] == 28.6
    assert metrics["tg128.samples_tokens_per_second.0"] == 28.1


def test_parse_markdown_bench_metrics():
    stdout = """
| model | test | t/s |
| --- | --- | ---: |
| gemma | tg128 | 39.25 |
| gemma | pp512 | 210.5 |
"""

    metrics = parse_llama_bench_metrics(stdout)

    assert metrics["tg128.tokens_per_second"] == 39.25
    assert metrics["pp512.tokens_per_second"] == 210.5
