import json
import os

from llm_refinery.benchmarks.lm_eval.command import build_lm_eval_command
from llm_refinery.benchmarks.lm_eval.config import LmEvalConfig
from llm_refinery.benchmarks.lm_eval.parser import latest_lm_eval_result, parse_lm_eval_metrics
from llm_refinery.core.endpoints import OPENAI_CHAT, Endpoint


def test_lm_eval_command_supports_include_path_for_fixed_tasks(tmp_path):
    config = LmEvalConfig(tasks="gpqa_main_generative_n_shot", include_path=tmp_path)
    cmd = build_lm_eval_command(
        config,
        Endpoint(
            name="llama_cpp",
            protocol=OPENAI_CHAT,
            model="local-model",
            base_url="http://localhost/v1/chat/completions",
        ),
    )

    assert "--include_path" in cmd
    assert str(tmp_path) in cmd


def test_latest_lm_eval_result_ignores_stale_result_files(tmp_path):
    old_result = tmp_path / "results_older.json"
    old_result.write_text("{}", encoding="utf-8")
    os.utime(old_result, (100.0, 100.0))

    assert latest_lm_eval_result(tmp_path, newer_than=200.0) is None

    new_result = tmp_path / "results_newer.json"
    new_result.write_text("{}", encoding="utf-8")
    os.utime(new_result, (300.0, 300.0))

    assert latest_lm_eval_result(tmp_path, newer_than=200.0) == new_result


def test_parse_lm_eval_metrics_normalizes_filters_and_stderr(tmp_path):
    result_path = tmp_path / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "results": {
                    "gsm8k": {
                        "exact_match,strict-match": 0.8143,
                        "exact_match_stderr,strict-match": 0.0107,
                        "exact_match,flexible-extract": 0.834,
                        "alias": "gsm8k",
                    },
                    "ifeval": {
                        "prompt_strict_acc": 0.8854,
                        "prompt_strict_acc_stderr": 0.012,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    assert parse_lm_eval_metrics(result_path) == {
        "gsm8k.strict-match.exact_match": 0.8143,
        "gsm8k.strict-match.exact_match_stderr": 0.0107,
        "gsm8k.flexible-extract.exact_match": 0.834,
        "ifeval.prompt_strict_acc": 0.8854,
        "ifeval.prompt_strict_acc_stderr": 0.012,
    }
