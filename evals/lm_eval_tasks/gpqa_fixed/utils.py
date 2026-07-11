from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datasets


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")

    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = [
            (preprocess(doc["Incorrect Answer 1"]), False),
            (preprocess(doc["Incorrect Answer 2"]), False),
            (preprocess(doc["Incorrect Answer 3"]), False),
            (preprocess(doc["Correct Answer"]), True),
        ]
        seed_material = "\0".join(
            [
                str(doc.get("Question") or ""),
                *(choice for choice, _is_correct in choices),
            ]
        )
        seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8])
        random.Random(seed).shuffle(choices)
        correct_answer_index = next(
            index for index, (_choice, is_correct) in enumerate(choices) if is_correct
        )
        choice_texts = [choice for choice, _is_correct in choices]

        out_doc = {
            "choice1": choice_texts[0],
            "choice2": choice_texts[1],
            "choice3": choice_texts[2],
            "choice4": choice_texts[3],
            "choices": choice_texts,
            "answer": f"({chr(65 + correct_answer_index)})",
        }
        return out_doc

    return dataset.map(_process_doc)
