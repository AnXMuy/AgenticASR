from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from benchmark_io import JsonObject, JsonValue


class PostprocessError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class InferenceOutcome:
    clean_text: str | None
    llm_latency_ms: float
    error: str | None


def _output_object(record: JsonObject) -> JsonObject:
    output = record.get("output")
    if not isinstance(output, dict):
        raise PostprocessError("Record is missing an output object")
    return output


def extract_raw_text(record: JsonObject) -> str:
    raw_text = _output_object(record).get("raw_text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise PostprocessError("Record is missing non-empty output.raw_text")
    return raw_text


def source_record_id(record: JsonObject) -> str:
    value = record.get("source_record_id")
    if not isinstance(value, str) or not value.strip():
        raise PostprocessError("Record is missing non-empty source_record_id")
    return value


def _latency(value: JsonValue) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def build_result_record(source: JsonObject, outcome: InferenceOutcome) -> JsonObject:
    result = copy.deepcopy(source)
    source_output = _output_object(source)
    raw_value = source_output.get("raw_text")
    raw_text = raw_value if isinstance(raw_value, str) else ""
    stt_latency = _latency(source_output.get("stt_latency_ms"))
    total_latency = (
        stt_latency + outcome.llm_latency_ms if stt_latency is not None else None
    )
    result["output"] = {
        "clean_text": outcome.clean_text or "",
        "error": outcome.error,
        "llm_latency_ms": outcome.llm_latency_ms,
        "ok": outcome.error is None,
        "raw_text": raw_text,
        "stt_latency_ms": stt_latency,
        "total_latency_ms": total_latency,
    }
    return result


def completed_record_ids(rows: list[JsonObject]) -> set[str]:
    latest: dict[str, bool] = {}
    for row in rows:
        record_id = row.get("source_record_id")
        output = row.get("output")
        if isinstance(record_id, str) and isinstance(output, dict):
            error = output.get("error")
            latest[record_id] = output.get("ok") is True or (
                isinstance(error, str) and error.startswith("input_error:")
            )
    return {record_id for record_id, succeeded in latest.items() if succeeded}
