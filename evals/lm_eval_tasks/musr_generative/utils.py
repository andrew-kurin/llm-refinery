from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict[str, Any]) -> dict[str, str]:
        choices = [str(choice) for choice in ast.literal_eval(str(doc["choices"]))]
        answer = str(doc["answer_choice"])
        if answer not in choices:
            raise ValueError("MuSR answer_choice is not present in choices")
        return {
            "choice_lines": "\n".join(
                f"{index} - {choice}" for index, choice in enumerate(choices, start=1)
            ),
            "answer_label": str(choices.index(answer) + 1),
        }

    return dataset.map(_process_doc)


def doc_to_text(doc: dict[str, Any]) -> str:
    return (
        f"{doc['narrative']}\n\n{doc['question']}\n\n{doc['choice_lines']}\n\n"
        "Respond with only the number of the correct choice.\nAnswer:"
    )
