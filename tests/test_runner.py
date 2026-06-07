from llama_tune.runner import BenchProgress, format_duration


def test_format_duration():
    assert format_duration(0) == "0s"
    assert format_duration(0.25) == "<1s"
    assert format_duration(9.4) == "9s"
    assert format_duration(65) == "1m05s"
    assert format_duration(3661) == "1h01m01s"


def test_bench_progress_estimates_from_completed_trials():
    progress = BenchProgress(total=4, started_monotonic=0.0)

    assert progress.eta_after_completed_s() is None
    assert progress.eta_during_current_s(3.0) is None

    progress.record_completion(10.0)

    assert progress.average_duration_s == 10.0
    assert progress.eta_after_completed_s() == 30.0
    assert progress.eta_during_current_s(4.0) == 26.0
