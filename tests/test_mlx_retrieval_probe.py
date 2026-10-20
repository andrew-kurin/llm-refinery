import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "mlx_retrieval_probe", Path(__file__).resolve().parents[1] / "tools" / "mlx_retrieval_probe.py"
)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("position", ["start", "middle", "near-end"])
@pytest.mark.parametrize("task", ["single", "pair", "revision"])
def test_varied_retrieval_has_one_answer_occurrence_and_exact_length(position, task):
    # Whitespace counts keep this invariant test independent of MLX and downloads.
    case = probe.retrieval_case(lambda text: len(text.split()), 512, position, task, 601)
    assert len(case["prompt"].split()) == 512
    answers = case["expected_exact"].split("|")
    assert len(answers) == (2 if task == "pair" else 1)
    assert all(case["prompt"].count(answer) == 1 for answer in answers)
    if task == "revision":
        assert "APPROVED revision 7" in case["prompt"]
        assert "superseded revision 6" in case["prompt"]
    if position == "start":
        assert case["prompt"].index(answers[0]) < 150
    else:
        assert case["prompt"].startswith("Log 0000:")


def test_varied_retrieval_is_reproducible_and_changes_with_fixture_seed():
    count = lambda text: len(text.split())  # noqa: E731
    first = probe.retrieval_case(count, 512, "middle", "pair", 601)
    assert probe.retrieval_case(count, 512, "middle", "pair", 601) == first
    assert probe.retrieval_case(count, 512, "middle", "pair", 602) != first
