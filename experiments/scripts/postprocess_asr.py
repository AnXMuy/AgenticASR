#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark_io import JsonObject, read_jsonl
from postprocess_contract import (
    InferenceOutcome,
    PostprocessError,
    build_result_record,
    completed_record_ids,
    extract_raw_text,
    source_record_id,
)

SYSTEM_PROMPT = (
    "你是 ASR 文本纠错助手。保留原意，最小修改：去口癖/重复，修错字，补必要标点，"
    "规范数字、日期、术语和代码符号，处理自我修正。不要总结、扩写或解释。"
    "重要易错实体在末尾追加 <KEY>[词1、词2]；没有则不加。"
)
Conversation: TypeAlias = list[dict[str, str]]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_path: str
    device_map: str
    dtype: str
    trust_remote_code: bool
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float


@dataclass(frozen=True, slots=True)
class RunConfig:
    input_path: Path
    output_path: Path
    batch_size: int
    limit: int | None
    overwrite: bool
    validate_only: bool
    model: ModelConfig


def build_conversations(raw_texts: list[str]) -> list[Conversation]:
    return [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ]
        for raw_text in raw_texts
    ]


def thinking_template_kwargs(template: str) -> dict[str, bool]:
    return {"enable_thinking": False} if "enable_thinking" in template else {}


class TransformersPostprocessor:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            padding_side="left",
            trust_remote_code=config.trust_remote_code,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            dtype=config.dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
        ).eval()
        template = self._tokenizer.get_chat_template()
        self._template_kwargs = thinking_template_kwargs(template)

    def generate(self, raw_texts: list[str]) -> tuple[list[str], float]:
        conversations = build_conversations(raw_texts)
        inputs = self._tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
            **self._template_kwargs,
        )
        input_device = self._model.get_input_embeddings().weight.device
        inputs = inputs.to(input_device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        generation_args = {
            "max_new_tokens": self._config.max_new_tokens,
            "do_sample": self._config.do_sample,
        }
        if self._config.do_sample:
            generation_args.update(
                {
                    "temperature": self._config.temperature,
                    "top_p": self._config.top_p,
                }
            )
        with torch.inference_mode():
            generated = self._model.generate(**inputs, **generation_args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        input_width = inputs["input_ids"].shape[1]
        generated_texts = self._tokenizer.batch_decode(
            generated[:, input_width:], skip_special_tokens=True
        )
        if len(generated_texts) != len(raw_texts):
            raise PostprocessError("Model returned a different number of outputs than inputs")
        return [text.strip() for text in generated_texts], elapsed_ms


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Post-process ASR raw_text with a local LM.")
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return RunConfig(
        input_path=args.input_jsonl.resolve(),
        output_path=args.output_jsonl.resolve(),
        batch_size=args.batch_size,
        limit=args.limit,
        overwrite=args.overwrite,
        validate_only=args.validate_only,
        model=ModelConfig(
            model_path=args.model,
            device_map=args.device_map,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
    )


def _append_records(path: Path, records: list[JsonObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _pending_records(config: RunConfig) -> list[JsonObject]:
    if config.input_path == config.output_path:
        raise PostprocessError("Input and output paths must be different")
    input_records = read_jsonl(config.input_path)
    existing = (
        read_jsonl(config.output_path)
        if config.output_path.exists() and not config.overwrite
        else []
    )
    completed = completed_record_ids(existing)
    seen: set[str] = set()
    pending: list[JsonObject] = []
    for record in input_records:
        record_id = source_record_id(record)
        if record_id in seen:
            raise PostprocessError(f"Duplicate source_record_id in input: {record_id}")
        seen.add(record_id)
        if record_id not in completed:
            pending.append(record)
    return pending[: config.limit] if config.limit is not None else pending


def run(config: RunConfig) -> int:
    if config.input_path == config.output_path:
        raise PostprocessError("Input and output paths must be different")
    if config.overwrite and not config.validate_only and config.output_path.exists():
        config.output_path.unlink()
    pending = _pending_records(config)
    valid_records: list[JsonObject] = []
    invalid_records: list[JsonObject] = []
    for record in pending:
        try:
            extract_raw_text(record)
        except PostprocessError:
            invalid_records.append(record)
        else:
            valid_records.append(record)
    if config.validate_only:
        print(
            f"Validated input; {len(valid_records)} records ready, "
            f"{len(invalid_records)} records have no raw_text."
        )
        return 0
    if not pending:
        print("No pending records.")
        return 0
    processor = TransformersPostprocessor(config.model) if valid_records else None
    for offset in range(0, len(pending), config.batch_size):
        batch = pending[offset : offset + config.batch_size]
        batch_raw_texts: list[str] = []
        for record in batch:
            try:
                batch_raw_texts.append(extract_raw_text(record))
            except PostprocessError:
                continue
        generated_texts: list[str] = []
        latency_ms = 0.0
        if batch_raw_texts:
            if processor is None:
                raise PostprocessError("Postprocessor was not initialized")
            generated_texts, latency_ms = processor.generate(batch_raw_texts)
        generated_iterator = iter(generated_texts)
        outcomes: list[InferenceOutcome] = []
        for record in batch:
            try:
                extract_raw_text(record)
            except PostprocessError:
                outcomes.append(
                    InferenceOutcome(
                        None,
                        0.0,
                        "input_error: missing non-empty output.raw_text",
                    )
                )
            else:
                outcomes.append(InferenceOutcome(next(generated_iterator), latency_ms, None))
        _append_records(
            config.output_path,
            [build_result_record(record, outcome) for record, outcome in zip(batch, outcomes)],
        )
        completed = min(offset + len(batch), len(pending))
        print(f"Completed {completed}/{len(pending)} records.")
    print(f"Output: {config.output_path}")
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (OSError, PostprocessError, json.JSONDecodeError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
