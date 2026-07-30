#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from benchmark_io import (
    DataError,
    load_rubric,
    match_records,
    read_jsonl,
)
from client import DEFAULT_MODEL, ClientConfig, GemmaJudgeClient, JudgeError
from result_store import (
    StoreConfig,
    append_rows,
    completed_sample_ids,
    evaluate_sample,
    latest_rows,
    write_summary,
)

@dataclass(frozen=True, slots=True)
class RunConfig:
    asr_path: Path
    rubric_path: Path
    results_path: Path
    summary_path: Path
    text_field: str | None
    workers: int
    limit: int | None
    overwrite: bool
    validate_only: bool
    client: ClientConfig
    sample_ids: frozenset[str] | None = None


def _chat_completions_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Score ASR JSONL output with the Gemma rubric judge."
    )
    parser.add_argument("asr_jsonl", type=Path)
    parser.add_argument(
        "--rubric",
        type=Path,
        required=True,
        help="path to the benchmark rubric.json downloaded from the benchmark release",
    )
    parser.add_argument("--results", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--text-field")
    parser.add_argument("--api-url", default=os.getenv("GEMMA_API_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("GEMMA_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("GEMMA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--workers", type=int, default=int(os.getenv("GEMMA_JUDGE_WORKERS", "4")))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("GEMMA_TIMEOUT", "120")))
    parser.add_argument("--attempts", type=int, default=int(os.getenv("GEMMA_ATTEMPTS", "3")))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    asr_path = args.asr_jsonl.resolve()
    default_stem = asr_path.with_suffix("")
    results_path = (args.results or Path(f"{default_stem}.gemma_judge.jsonl")).resolve()
    summary_path = (args.summary or Path(f"{default_stem}.gemma_judge.summary.json")).resolve()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return RunConfig(
        asr_path=asr_path,
        rubric_path=args.rubric.resolve(),
        results_path=results_path,
        summary_path=summary_path,
        text_field=args.text_field,
        workers=args.workers,
        limit=args.limit,
        overwrite=args.overwrite,
        validate_only=args.validate_only,
        client=ClientConfig(
            api_url=_chat_completions_url(args.api_url),
            model=args.model,
            api_key=args.api_key,
            timeout_seconds=args.timeout,
            max_attempts=args.attempts,
        ),
    )


def run(config: RunConfig) -> int:
    rubric_records = load_rubric(config.rubric_path)
    asr_records = read_jsonl(config.asr_path)
    samples = match_records(rubric_records, asr_records, config.text_field)
    out_of_benchmark_count = len(asr_records) - len(samples)
    if config.sample_ids is not None:
        samples = [sample for sample in samples if sample.rubric.sample_id in config.sample_ids]
    if config.limit is not None:
        samples = samples[: config.limit]
    if config.validate_only:
        failed_asr = sum(sample.output is None for sample in samples)
        print(
            f"Validated {len(samples)} evaluable samples "
            f"({failed_asr} ASR failures, {out_of_benchmark_count} "
            "out-of-benchmark ASR samples skipped)."
        )
        return 0
    if config.overwrite and config.results_path.exists():
        config.results_path.unlink()
    client = GemmaJudgeClient(config.client)
    latest = latest_rows(config.results_path)
    completed = completed_sample_ids(samples, latest, client.evaluation_identity)
    pending = [sample for sample in samples if sample.rubric.sample_id not in completed]
    print(f"Evaluating {len(pending)} samples; {len(completed)} already complete.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.workers) as executor:
        futures = {executor.submit(evaluate_sample, sample, client): sample for sample in pending}
        with tqdm(total=len(pending), desc="Judging", unit="sample") as progress:
            for future in concurrent.futures.as_completed(futures):
                rows = future.result()
                append_rows(config.results_path, rows)
                progress.update(1)
    write_summary(
        StoreConfig(
            asr_path=config.asr_path,
            rubric_path=config.rubric_path,
            results_path=config.results_path,
            summary_path=config.summary_path,
            text_field=config.text_field,
            judge_model=config.client.model,
            judge_identity=client.evaluation_identity,
            input_asr_sample_count=len(asr_records),
            out_of_benchmark_asr_sample_count=out_of_benchmark_count,
        ),
        samples,
    )
    print(f"Question results: {config.results_path}")
    print(f"Summary metrics: {config.summary_path}")
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (DataError, JudgeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
