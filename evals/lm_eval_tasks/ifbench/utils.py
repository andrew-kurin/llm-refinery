from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

OFFICIAL_IFBENCH_COMMIT = "1091c4c3de6c1f6ed12c012ed68f11ea450b0117"
OFFICIAL_DATASET_REVISION = "2e8a48de45ff3bf41242f927254ca81b59ca3ae2"
OFFICIAL_IFBENCH_SPEC = (
    f"ifbench @ git+https://github.com/allenai/IFBench.git@{OFFICIAL_IFBENCH_COMMIT}"
)

# The upstream package does not expose its source revision at runtime. Hash the
# four scoring modules so an accidentally unpinned or modified grader cannot
# silently produce results under the same task name.
_EXPECTED_SOURCE_SHA256 = {
    "evaluation_lib": "33dba01a70d3ebca48341e9369f6ca9fc94d6e7da651a140fccece7ff3c34aaf",
    "instructions": "bf01c592df2e34430a602311addda4b06dbd1e69b3c49aea70f8edfe36bc581d",
    "instructions_registry": "4df83d050d29d8ccf8d476de1319d47d7b8379143e85d701c9c99d4816c19cd9",
    "instructions_util": "b7d60e07bbb2c56e42ee1a4a1b2f7281ced6c7af3af2719d2dca322afd3c20b8",
}

# These are the scorer-affecting versions from the official repository's
# uv.lock at OFFICIAL_IFBENCH_COMMIT. The setuptools cap is required because
# syllapy imports pkg_resources, which setuptools 81+ no longer provides.
_EXPECTED_DISTRIBUTION_VERSIONS = {
    "ifbench": "0.1.0",
    "emoji": "2.15.0",
    "nltk": "3.9.2",
    "setuptools": "80.9.0",
    "syllapy": "0.7.2",
}


def _dependency_error(detail: str) -> RuntimeError:
    return RuntimeError(
        "IFBench's pinned official grader is unavailable or does not match the "
        f"expected source ({detail}). Install the exact dependencies from "
        "evals/lm_eval_tasks/ifbench/requirements.txt."
    )


@lru_cache(maxsize=1)
def _load_official_grader() -> tuple[ModuleType, Mapping[str, type[Any]]]:
    for distribution, expected in _EXPECTED_DISTRIBUTION_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise _dependency_error(f"missing {distribution}=={expected}") from exc
        if actual != expected:
            raise _dependency_error(
                f"{distribution}=={actual} installed; expected {distribution}=={expected}"
            )

    modules: dict[str, ModuleType] = {}
    for module_name, expected_digest in _EXPECTED_SOURCE_SHA256.items():
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError) as exc:
            raise _dependency_error(f"could not import {module_name}: {exc}") from exc

        source = getattr(module, "__file__", None)
        if not source:
            raise _dependency_error(f"{module_name} has no inspectable source file")
        try:
            actual_digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        except OSError as exc:
            raise _dependency_error(f"could not read {module_name} source: {exc}") from exc
        if actual_digest != expected_digest:
            raise _dependency_error(
                f"{module_name} source hash {actual_digest} does not match "
                f"commit {OFFICIAL_IFBENCH_COMMIT}"
            )
        modules[module_name] = module

    registry = getattr(modules["instructions_registry"], "INSTRUCTION_DICT", None)
    if not isinstance(registry, Mapping) or not registry:
        raise _dependency_error("official instruction registry is missing or empty")
    return modules["evaluation_lib"], registry


def _validated_doc(doc: Mapping[str, Any]) -> tuple[Any, str, list[str], list[dict[str, Any]]]:
    missing = {"key", "prompt", "instruction_id_list", "kwargs"}.difference(doc)
    if missing:
        raise ValueError(f"IFBench record is missing required fields: {sorted(missing)}")

    prompt = doc["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("IFBench record prompt must be a non-empty string")

    instruction_ids = doc["instruction_id_list"]
    if (
        not isinstance(instruction_ids, Sequence)
        or isinstance(instruction_ids, (str, bytes))
        or not instruction_ids
        or not all(isinstance(item, str) and item for item in instruction_ids)
    ):
        raise ValueError("IFBench instruction_id_list must be a non-empty list of strings")

    kwargs = doc["kwargs"]
    if (
        not isinstance(kwargs, Sequence)
        or isinstance(kwargs, (str, bytes))
        or len(kwargs) != len(instruction_ids)
        or not all(isinstance(item, Mapping) for item in kwargs)
    ):
        raise ValueError("IFBench kwargs must contain one mapping per instruction")

    return (
        doc["key"],
        prompt,
        list(instruction_ids),
        # The Hub's Parquet conversion represents the union of every kwargs
        # field and fills irrelevant fields with null. The official JSONL is
        # sparse, and its loose scorer does not remove those null fields before
        # dispatching to checker-specific build_description methods.
        [{key: value for key, value in item.items() if value is not None} for item in kwargs],
    )


def _score_mode(
    evaluation_lib: ModuleType,
    *,
    mode: str,
    key: Any,
    prompt: str,
    instruction_ids: list[str],
    kwargs: list[dict[str, Any]],
    response: str,
) -> tuple[bool, list[bool]]:
    example = evaluation_lib.InputExample(
        key=key,
        instruction_id_list=copy.deepcopy(instruction_ids),
        prompt=prompt,
        kwargs=copy.deepcopy(kwargs),
    )
    scorer = getattr(evaluation_lib, f"test_instruction_following_{mode}", None)
    if not callable(scorer):
        raise _dependency_error(f"official {mode} scorer is missing")
    output = scorer(example, {prompt: response})
    per_instruction = list(output.follow_instruction_list)
    if len(per_instruction) != len(instruction_ids) or not all(
        isinstance(item, bool) for item in per_instruction
    ):
        raise RuntimeError(f"Official IFBench {mode} scorer returned an invalid result")
    return bool(output.follow_all_instructions), per_instruction


def process_results(doc: Mapping[str, Any], results: Sequence[Any]) -> dict[str, Any]:
    if len(results) != 1 or not isinstance(results[0], str):
        raise ValueError("IFBench expects exactly one text generation per record")

    evaluation_lib, registry = _load_official_grader()
    key, prompt, instruction_ids, kwargs = _validated_doc(doc)
    unknown = sorted(set(instruction_ids).difference(registry))
    if unknown:
        raise ValueError(f"IFBench record contains unknown instruction IDs: {unknown}")

    strict_prompt, strict_instructions = _score_mode(
        evaluation_lib,
        mode="strict",
        key=key,
        prompt=prompt,
        instruction_ids=instruction_ids,
        kwargs=kwargs,
        response=results[0],
    )
    loose_prompt, loose_instructions = _score_mode(
        evaluation_lib,
        mode="loose",
        key=key,
        prompt=prompt,
        instruction_ids=instruction_ids,
        kwargs=kwargs,
        response=results[0],
    )

    return {
        "prompt_level_loose_acc": float(loose_prompt),
        "inst_level_loose_acc": (sum(loose_instructions), len(loose_instructions)),
        "prompt_level_strict_acc": float(strict_prompt),
        "inst_level_strict_acc": (sum(strict_instructions), len(strict_instructions)),
    }


def aggregate_instruction_accuracy(items: Sequence[tuple[int, int]]) -> float:
    correct = sum(correct for correct, _ in items)
    total = sum(total for _, total in items)
    if total <= 0:
        raise ValueError("Cannot aggregate IFBench instruction accuracy without instructions")
    return correct / total
