from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from benchmark_io import JsonObject, JsonValue, MatchedSample, read_jsonl
from client import GemmaJudgeClient, JudgeError
from metrics import summarize, summarize_asr

TERMINAL_STATUSES: Final = {"success"}


@dataclass(frozen=True, slots=True)
class StoreConfig:
    asr_path: Path
    rubric_path: Path
    results_path: Path
    summary_path: Path
    text_field: str | None
    judge_model: str
    judge_identity: str
    input_asr_sample_count: int
    out_of_benchmark_asr_sample_count: int


def _evaluation_signature(
    sample: MatchedSample, question_index: int, judge_identity: str
) -> str:
    question = sample.rubric.questions[question_index]
    value = {
        "judge_model": judge_identity,
        "oral": sample.rubric.oral,
        "clean": sample.rubric.clean,
        "output": sample.output or "",
        "question": question.text,
        "evidence": question.evidence,
        "category": question.category,
        "options": [
            {"text": option.text, "score": option.score} for option in question.options
        ],
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_signature(sample: MatchedSample, question_index: int) -> str:
    question = sample.rubric.questions[question_index]
    value = {
        "oral": sample.rubric.oral,
        "clean": sample.rubric.clean,
        "output": sample.output or "",
        "question": question.text,
        "evidence": question.evidence,
        "category": question.category,
        "options": [
            {"text": option.text, "score": option.score} for option in question.options
        ],
    }
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _base_row(
    sample: MatchedSample,
    question_index: int,
    client: GemmaJudgeClient,
) -> JsonObject:
    question = sample.rubric.questions[question_index]
    return {
        "sample_id": sample.rubric.sample_id,
        "audio": sample.rubric.audio,
        "scene": sample.rubric.scene,
        "question_id": question.question_id,
        "category": question.category,
        "question": question.text,
        "evidence": question.evidence,
        "oral": sample.rubric.oral,
        "clean": sample.rubric.clean,
        "output": sample.output or "",
        "judge_model": client.model,
        "content_signature": _content_signature(sample, question_index),
        "evaluation_signature": _evaluation_signature(
            sample, question_index, client.evaluation_identity
        ),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_sample(sample: MatchedSample, client: GemmaJudgeClient) -> list[JsonObject]:
    try:
        answers = client.judge(sample.rubric, sample.output)
    except JudgeError:
        # Do not create a partial aggregate that excludes failed judge questions.
        # The caller aborts and the run can be retried as a whole.
        raise
    answer_by_id = {answer.question_id: answer.option_index for answer in answers}
    scored_rows: list[JsonObject] = []
    for index, question in enumerate(sample.rubric.questions):
        option_index = answer_by_id[question.question_id]
        option = question.options[option_index]
        row = _base_row(sample, index, client)
        row.update(
            {
                "status": "success",
                "selected_option_index": option_index,
                "selected_option": option.text,
                "score": option.score,
                "max_score": max(item.score for item in question.options),
            }
        )
        if sample.asr_error is not None:
            row["asr_retry_exhausted"] = True
            row["asr_error"] = sample.asr_error
        scored_rows.append(row)
    return scored_rows


def latest_rows(path: Path) -> dict[tuple[str, str], JsonObject]:
    if not path.exists():
        return {}
    latest: dict[tuple[str, str], JsonObject] = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        question_id = row.get("question_id")
        if isinstance(sample_id, str) and isinstance(question_id, str):
            latest[(sample_id, question_id)] = row
    return latest


def completed_sample_ids(
    samples: list[MatchedSample],
    latest: dict[tuple[str, str], JsonObject],
    judge_identity: str,
) -> set[str]:
    completed: set[str] = set()
    for sample in samples:
        statuses: list[JsonValue] = []
        for index, question in enumerate(sample.rubric.questions):
            row = latest.get((sample.rubric.sample_id, question.question_id), {})
            expected = _evaluation_signature(sample, index, judge_identity)
            statuses.append(row.get("status") if row.get("evaluation_signature") == expected else None)
        if statuses and all(status in TERMINAL_STATUSES for status in statuses):
            completed.add(sample.rubric.sample_id)
    return completed


def append_rows(path: Path, rows: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_summary(config: StoreConfig, samples: list[MatchedSample]) -> None:
    latest = latest_rows(config.results_path)
    current_rows: dict[tuple[str, str], JsonObject] = {}
    for sample in samples:
        for index, question in enumerate(sample.rubric.questions):
            key = (sample.rubric.sample_id, question.question_id)
            row = latest.get(key)
            expected = _evaluation_signature(sample, index, config.judge_identity)
            if row is not None and row.get("evaluation_signature") == expected:
                current_rows[key] = row
    completed = completed_sample_ids(samples, current_rows, config.judge_identity)
    summary = summarize(
        list(current_rows.values()),
        total_samples=len(samples),
        completed_samples=len(completed),
    )
    summary["asr_metrics"] = summarize_asr(samples)
    run_metrics = summary["run"]
    if isinstance(run_metrics, dict):
        run_metrics["input_asr_sample_count"] = config.input_asr_sample_count
        run_metrics["out_of_benchmark_asr_sample_count"] = (
            config.out_of_benchmark_asr_sample_count
        )
    summary["config"] = {
        "asr_jsonl": str(config.asr_path),
        "rubric": str(config.rubric_path),
        "results": str(config.results_path),
        "model": config.judge_model,
        "text_field": config.text_field or "auto",
        "weighting": "question",
    }
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.summary_path.with_suffix(config.summary_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(config.summary_path)
