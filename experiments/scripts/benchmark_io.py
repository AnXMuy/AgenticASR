from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]
STRING_OPTION_PATTERN = re.compile(
    r"^\s*(.+?)\s*[\uff08(]\s*(-?\d+)\s*[\uff09)]\s*$"
)
JOINED_OPTIONS_PATTERN = re.compile(
    r"(?:^|[\uff0c,])\s*(.+?)\s*[\uff08(]\s*(-?\d+)\s*[\uff09)]"
    r"(?=\s*[\uff0c,]|\s*$)"
)


class DataError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class Option:
    text: str
    score: int


@dataclass(frozen=True, slots=True)
class Question:
    question_id: str
    category: str
    text: str
    evidence: str
    options: tuple[Option, ...]


@dataclass(frozen=True, slots=True)
class RubricSample:
    sample_id: str
    audio: str
    scene: str
    oral: str
    clean: str
    questions: tuple[Question, ...]


@dataclass(frozen=True, slots=True)
class MatchedSample:
    rubric: RubricSample
    output: str | None
    latency_ms: float | None
    asr_error: str | None


def read_jsonl(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise DataError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def load_rubric(path: Path) -> list[JsonObject]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DataError(f"Invalid rubric JSON at {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("items"), list):
        raise DataError(f"Rubric must contain an items array: {path}")
    items = value["items"]
    if not all(isinstance(item, dict) for item in items):
        raise DataError(f"Every rubric item must be an object: {path}")
    return items


def _required_string(record: JsonObject, field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DataError(f"Missing non-empty {field} in {context}")
    return value


def _parse_question(value: JsonValue, sample_id: str) -> Question:
    if not isinstance(value, dict):
        raise DataError(f"Invalid QA object in {sample_id}")
    question_id = _required_string(value, "id", sample_id)
    category = _required_string(value, "category", f"{sample_id}/{question_id}")
    if category not in {"content", "format", "filter", "rephrase"}:
        raise DataError(f"Invalid category {category} in {sample_id}/{question_id}")
    raw_options = value.get("options")
    if isinstance(raw_options, str):
        matches = list(JOINED_OPTIONS_PATTERN.finditer(raw_options))
        unmatched = JOINED_OPTIONS_PATTERN.sub("", raw_options).strip(" \uff0c,")
        if unmatched:
            raise DataError(f"Invalid joined options in {sample_id}/{question_id}")
        joined_options = [
            {"text": match.group(1), "score": int(match.group(2))}
            for match in matches
        ]
        raw_options = joined_options
    if not isinstance(raw_options, list) or not raw_options:
        raise DataError(f"Missing options in {sample_id}/{question_id}")
    options: list[Option] = []
    for raw_option in raw_options:
        if isinstance(raw_option, str):
            match = STRING_OPTION_PATTERN.fullmatch(raw_option)
            if match is None:
                raise DataError(f"Invalid string option in {sample_id}/{question_id}")
            options.append(Option(match.group(1), int(match.group(2))))
            continue
        if not isinstance(raw_option, dict):
            raise DataError(f"Invalid option in {sample_id}/{question_id}")
        score = raw_option.get("score")
        if not isinstance(score, int):
            raise DataError(f"Invalid option score in {sample_id}/{question_id}")
        options.append(Option(_required_string(raw_option, "text", question_id), score))
    return Question(
        question_id=question_id,
        category=category,
        text=_required_string(value, "question", f"{sample_id}/{question_id}"),
        evidence=_required_string(value, "evidence", f"{sample_id}/{question_id}"),
        options=tuple(options),
    )


def parse_rubric_sample(record: JsonObject) -> RubricSample:
    sample_id = _required_string(record, "source_record_id", "rubric item")
    raw_questions = record.get("qa")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise DataError(f"Missing QA list in {sample_id}")
    questions = tuple(_parse_question(item, sample_id) for item in raw_questions)
    question_ids = {question.question_id for question in questions}
    if len(question_ids) != len(questions):
        raise DataError(f"Duplicate question id in {sample_id}")
    return RubricSample(
        sample_id=sample_id,
        audio=_required_string(record, "audio", sample_id),
        scene=_required_string(record, "scene", sample_id),
        oral=_required_string(record, "oral", sample_id),
        clean=_required_string(record, "clean", sample_id),
        questions=questions,
    )


def _nested_value(record: JsonObject, field: str) -> JsonValue:
    current: JsonValue = record
    for part in field.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def extract_asr_text(record: JsonObject, text_field: str | None) -> str | None:
    if text_field:
        value = _nested_value(record, text_field)
        return value.strip() if isinstance(value, str) and value.strip() else None
    nested = record.get("output")
    if isinstance(nested, dict):
        clean_text = nested.get("clean_text")
        if nested.get("ok") is True and isinstance(clean_text, str) and clean_text.strip():
            return clean_text.strip()
    formal_text = record.get("formalasr_text")
    if record.get("formalasr_status") == "ok" and isinstance(formal_text, str):
        return formal_text.strip() or None
    return None


def _latency_value(value: JsonValue) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    latency = float(value)
    return latency if math.isfinite(latency) and latency >= 0 else None


def extract_asr_latency_ms(record: JsonObject) -> float | None:
    nested = record.get("output")
    if isinstance(nested, dict):
        total = _latency_value(nested.get("total_latency_ms"))
        if total is not None:
            return total
    for field in ("total_latency_ms", "formalasr_latency_ms"):
        latency = _latency_value(record.get(field))
        if latency is not None:
            return latency
    if isinstance(nested, dict):
        stt = _latency_value(nested.get("stt_latency_ms"))
        llm = _latency_value(nested.get("llm_latency_ms"))
        if stt is not None and llm is not None:
            return stt + llm
    return _latency_value(record.get("latency_ms"))


def _asr_error(record: JsonObject) -> str:
    nested = record.get("output")
    if isinstance(nested, dict) and isinstance(nested.get("error"), str):
        return nested["error"]
    formal_error = record.get("formalasr_error_message")
    if isinstance(formal_error, str) and formal_error.strip():
        return formal_error
    return "ASR output is empty or unsuccessful"


def match_records(
    rubric_records: list[JsonObject],
    asr_records: list[JsonObject],
    text_field: str | None = None,
) -> list[MatchedSample]:
    asr_by_id: dict[str, JsonObject] = {}
    for record in asr_records:
        sample_id = _required_string(record, "source_record_id", "ASR record")
        if sample_id in asr_by_id:
            raise DataError(f"Duplicate ASR source_record_id: {sample_id}")
        asr_by_id[sample_id] = record
    rubric_by_id: dict[str, JsonObject] = {}
    for record in rubric_records:
        sample_id = _required_string(record, "source_record_id", "rubric item")
        if sample_id in rubric_by_id:
            raise DataError(f"Duplicate rubric source_record_id: {sample_id}")
        rubric_by_id[sample_id] = record
    matched: list[MatchedSample] = []
    for sample_id in asr_by_id:
        if sample_id not in rubric_by_id:
            continue
        rubric_record = rubric_by_id[sample_id]
        if rubric_record.get("status") != "success" or not rubric_record.get("qa"):
            continue
        sample = parse_rubric_sample(rubric_record)
        asr_record = asr_by_id[sample.sample_id]
        output = extract_asr_text(asr_record, text_field)
        asr_error = None if output is not None else _asr_error(asr_record)
        matched.append(
            MatchedSample(
                rubric=sample,
                # Upstream ASR runners retry before writing this record. A request
                # still missing after retries is scored as an empty hypothesis,
                # never silently removed from the benchmark denominator.
                output=output if output is not None else "",
                latency_ms=extract_asr_latency_ms(asr_record),
                asr_error=asr_error,
            )
        )
    return matched
