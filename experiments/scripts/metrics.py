from __future__ import annotations

from benchmark_io import JsonObject, MatchedSample
from text_metrics import compute_text_metrics



class _Accumulator:
    __slots__ = ("max_possible_score", "question_count", "score_sum")

    def __init__(self) -> None:
        self.question_count = 0
        self.score_sum = 0
        self.max_possible_score = 0

    def add(self, score: int, max_score: int) -> None:
        self.question_count += 1
        self.score_sum += score
        self.max_possible_score += max_score

    def result(self) -> JsonObject:
        normalized_score = (
            self.score_sum / self.max_possible_score if self.max_possible_score else 0.0
        )
        return {
            "question_count": self.question_count,
            "total_score": self.score_sum,
            "max_possible_score": self.max_possible_score,
            "normalized_score": min(1.0, max(0.0, normalized_score)),
        }


def summarize(
    rows: list[JsonObject], total_samples: int, completed_samples: int
) -> JsonObject:
    overall = _Accumulator()
    by_category: dict[str, _Accumulator] = {}
    by_scene: dict[str, _Accumulator] = {}
    failed_questions = 0
    asr_retry_exhausted_questions = 0
    asr_retry_exhausted_samples: set[str] = set()
    for row in rows:
        status = row.get("status")
        sample_id = row.get("sample_id")
        if row.get("asr_retry_exhausted") is True:
            asr_retry_exhausted_questions += 1
            if isinstance(sample_id, str):
                asr_retry_exhausted_samples.add(sample_id)
        if status != "success":
            failed_questions += 1
            continue
        score = row.get("score")
        max_score = row.get("max_score")
        category = row.get("category")
        scene = row.get("scene")
        if not isinstance(score, int):
            continue
        if not isinstance(category, str) or not isinstance(scene, str):
            continue
        if not isinstance(max_score, int):
            max_score = 2 if category in {"filter", "rephrase"} else 1
        overall.add(score, max_score)
        by_category.setdefault(category, _Accumulator()).add(score, max_score)
        by_scene.setdefault(scene, _Accumulator()).add(score, max_score)
    run: JsonObject = {
        "total_sample_count": total_samples,
        "completed_sample_count": completed_samples,
        "failed_question_count": failed_questions,
        "asr_retry_exhausted_sample_count": len(asr_retry_exhausted_samples),
        "asr_retry_exhausted_question_count": asr_retry_exhausted_questions,
    }
    return {
        "run": run,
        "overall": overall.result(),
        "by_category": {
            key: accumulator.result() for key, accumulator in sorted(by_category.items())
        },
        "by_scene": {
            key: accumulator.result() for key, accumulator in sorted(by_scene.items())
        },
    }


def summarize_asr(samples: list[MatchedSample]) -> JsonObject:
    unique_samples = {sample.rubric.sample_id: sample for sample in samples}
    wer_errors = 0
    wer_reference_units = 0
    cer_errors = 0
    cer_reference_units = 0
    mer_errors = 0
    mer_reference_units = 0
    text_metric_sample_count = 0
    latencies: list[float] = []
    for sample in unique_samples.values():
        if sample.latency_ms is not None:
            latencies.append(sample.latency_ms)
        text_metrics = compute_text_metrics(sample.rubric.clean, sample.output or "")
        wer_errors += text_metrics.wer_errors
        wer_reference_units += text_metrics.wer_reference_units
        cer_errors += text_metrics.cer_errors
        cer_reference_units += text_metrics.cer_reference_units
        mer_errors += text_metrics.mer_errors
        mer_reference_units += text_metrics.mer_reference_units
        text_metric_sample_count += 1
    latency_sample_count = len(latencies)
    return {
        "sample_count": len(unique_samples),
        "text_metric_sample_count": text_metric_sample_count,
        "latency_sample_count": latency_sample_count,
        "missing_latency_sample_count": len(unique_samples) - latency_sample_count,
        "average_latency_ms": (
            sum(latencies) / latency_sample_count if latency_sample_count else 0.0
        ),
        "wer": wer_errors / wer_reference_units if wer_reference_units else 0.0,
        "wer_errors": wer_errors,
        "wer_reference_units": wer_reference_units,
        "cer": cer_errors / cer_reference_units if cer_reference_units else 0.0,
        "cer_errors": cer_errors,
        "cer_reference_units": cer_reference_units,
        "mer": mer_errors / mer_reference_units if mer_reference_units else 0.0,
        "mer_errors": mer_errors,
        "mer_reference_units": mer_reference_units,
    }
