import pytest

from llm_refinery.quality_compare import compare_paired_correctness


def _sample(sample_id, correct, task="ifeval"):
    return {
        "sample_id": sample_id,
        "payload_json": {"task": task},
        "metrics": {"correct": correct},
    }


def test_paired_quality_comparison_counts_flips_and_exact_significance():
    baseline = [
        _sample("a", 1),
        _sample("b", 1),
        _sample("c", 0),
        _sample("d", 0),
        _sample("baseline-only", 1),
    ]
    candidate = [
        _sample("a", 1),
        _sample("b", 0),
        _sample("c", 1),
        _sample("d", 1),
        _sample("candidate-only", 1),
    ]

    result = compare_paired_correctness(baseline, candidate)

    assert result.paired_count == 4
    assert result.baseline_only_count == 1
    assert result.candidate_only_count == 1
    assert result.candidate_win_count == 2
    assert result.candidate_loss_count == 1
    assert result.tie_count == 1
    assert result.accuracy_delta == 0.25
    assert result.mcnemar_exact_p == 1.0


def test_paired_quality_comparison_filters_task_and_requires_overlap():
    baseline = [_sample("ifeval:1", 1), _sample("gpqa:1", 0, task="gpqa")]
    candidate = [_sample("ifeval:1", 0), _sample("gpqa:2", 1, task="gpqa")]

    result = compare_paired_correctness(baseline, candidate, task="ifeval")
    assert result.paired_count == 1
    assert result.accuracy_delta == -1.0

    with pytest.raises(ValueError, match="no paired correctness"):
        compare_paired_correctness(baseline, candidate, task="gpqa")


def test_paired_quality_comparison_accepts_an_explicit_binary_sample_metric():
    baseline = [_sample("a", 0)]
    candidate = [_sample("a", 0)]
    baseline[0]["metrics"]["prompt_level_loose_acc"] = 0
    candidate[0]["metrics"]["prompt_level_loose_acc"] = 1

    result = compare_paired_correctness(
        baseline,
        candidate,
        sample_metric="prompt_level_loose_acc",
    )
    assert result.accuracy_delta == 1.0
