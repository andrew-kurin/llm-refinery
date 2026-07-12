from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TASK_DIR = Path(__file__).parents[1] / "evals" / "lm_eval_tasks" / "ifbench"


def _load_utils():
    spec = importlib.util.spec_from_file_location("refinery_ifbench_utils", TASK_DIR / "utils.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_instruction_accuracy_uses_micro_average():
    utils = _load_utils()

    assert utils.aggregate_instruction_accuracy([(1, 1), (1, 3)]) == 0.5


def test_ifbench_doc_validation_rejects_misaligned_instruction_arguments():
    utils = _load_utils()

    with pytest.raises(ValueError, match="one mapping per instruction"):
        utils._validated_doc(
            {
                "key": "example",
                "prompt": "Write a response.",
                "instruction_id_list": ["constraint:a", "constraint:b"],
                "kwargs": [{}],
            }
        )


def test_ifbench_doc_validation_restores_sparse_official_kwargs():
    utils = _load_utils()

    _, _, _, kwargs = utils._validated_doc(
        {
            "key": "example",
            "prompt": "Write a response.",
            "instruction_id_list": ["count:numbers"],
            "kwargs": [{"N": 2, "keyword": None}],
        }
    )

    assert kwargs == [{"N": 2}]


def test_ifbench_task_pins_dataset_and_official_scorer():
    utils = _load_utils()
    task = (TASK_DIR / "ifbench.yaml").read_text(encoding="utf-8")
    requirements = (TASK_DIR / "requirements.txt").read_text(encoding="utf-8")

    assert utils.OFFICIAL_DATASET_REVISION in task
    assert utils.OFFICIAL_IFBENCH_COMMIT in task
    assert utils.OFFICIAL_IFBENCH_COMMIT in requirements
