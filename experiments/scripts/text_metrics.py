from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final, Sequence, TypeVar

from opencc import OpenCC

_CJK: Final = r"\u3400-\u4dbf\u4e00-\u9fff"
_WORD_PATTERN: Final = re.compile(rf"[{_CJK}]+|[a-z0-9]+")
_MER_PATTERN: Final = re.compile(rf"[{_CJK}]|[a-z0-9]+")
_OPENCC: Final = OpenCC("t2s")
Token = TypeVar("Token", str, int)


@dataclass(frozen=True, slots=True)
class TextMetrics:
    wer_errors: int
    wer_reference_units: int
    cer_errors: int
    cer_reference_units: int
    mer_errors: int
    mer_reference_units: int

    @staticmethod
    def _rate(errors: int, reference_units: int) -> float:
        if reference_units:
            return errors / reference_units
        return 0.0 if errors == 0 else 1.0

    @property
    def wer(self) -> float:
        return self._rate(self.wer_errors, self.wer_reference_units)

    @property
    def cer(self) -> float:
        return self._rate(self.cer_errors, self.cer_reference_units)

    @property
    def mer(self) -> float:
        return self._rate(self.mer_errors, self.mer_reference_units)


def normalize_text(text: str) -> str:
    converted = _OPENCC.convert(unicodedata.normalize("NFKC", text)).casefold()
    normalized = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S", "Z"} else character
        for character in converted
    )
    return " ".join(normalized.split())


def _edit_distance(reference: Sequence[Token], hypothesis: Sequence[Token]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_token in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis, start=1):
            substitution_cost = 0 if reference_token == hypothesis_token else 1
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def compute_text_metrics(reference: str, hypothesis: str) -> TextMetrics:
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    reference_words = _WORD_PATTERN.findall(normalized_reference)
    hypothesis_words = _WORD_PATTERN.findall(normalized_hypothesis)
    reference_characters = list(normalized_reference.replace(" ", ""))
    hypothesis_characters = list(normalized_hypothesis.replace(" ", ""))
    reference_mixed = _MER_PATTERN.findall(normalized_reference)
    hypothesis_mixed = _MER_PATTERN.findall(normalized_hypothesis)
    return TextMetrics(
        wer_errors=_edit_distance(reference_words, hypothesis_words),
        wer_reference_units=len(reference_words),
        cer_errors=_edit_distance(reference_characters, hypothesis_characters),
        cer_reference_units=len(reference_characters),
        mer_errors=_edit_distance(reference_mixed, hypothesis_mixed),
        mer_reference_units=len(reference_mixed),
    )
