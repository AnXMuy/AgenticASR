"""Offline and sliding-window transcript refinement."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from .chunking import Chunk

SYSTEM_PROMPT = (
    "你是 ASR 文本纠错助手。保留原意，最小修改：去口癖/重复，修错字，补必要标点，"
    "规范数字、日期、术语和代码符号，处理自我修正。不要总结、扩写或解释。"
    "重要易错实体在末尾追加 <KEY>[词1、词2]；没有则不加。"
)

class TextRefiner(Protocol):
    """Backend contract used by offline and streaming inference."""

    def refine(self, text: str) -> str:
        """Refine one complete transcript or concatenated source window."""


@dataclass(frozen=True, slots=True)
class RefinementUpdate:
    """A replacement operation for one streaming window."""

    start_index: int
    raw_chunks: tuple[str, ...]
    refined_text: str
    transcript: str
    latency_ms: float


class StreamingRefinementSession:
    """Replace the active source window with one refined text string."""

    def __init__(self, refiner: TextRefiner, window_size: int = 3) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.refiner = refiner
        self.window_size = window_size
        self.raw_chunks: list[str] = []
        self._committed_text: list[str] = []
        self._active_text = ""

    @property
    def transcript(self) -> str:
        return _join_chunks([*self._committed_text, self._active_text])

    def add(self, chunk: Chunk | str) -> RefinementUpdate:
        text = chunk.text if isinstance(chunk, Chunk) else chunk
        text = text.strip()
        if not text:
            raise ValueError("chunk text must not be empty")

        self.raw_chunks.append(text)
        start = max(0, len(self.raw_chunks) - self.window_size)
        raw_window = self.raw_chunks[start:]

        started = time.perf_counter()
        while len(self._committed_text) < start:
            index = len(self._committed_text)
            self._committed_text.append(self._refine(self.raw_chunks[index]))
        refined_text = self._refine(_join_chunks(raw_window))
        latency_ms = (time.perf_counter() - started) * 1000
        self._active_text = refined_text
        return RefinementUpdate(
            start_index=start,
            raw_chunks=tuple(raw_window),
            refined_text=refined_text,
            transcript=self.transcript,
            latency_ms=latency_ms,
        )

    def _refine(self, text: str) -> str:
        refined = self.refiner.refine(text)
        if not isinstance(refined, str):
            raise ValueError("refiner must return one text string")
        return refined.strip()


class IdentityRefiner:
    """No-op backend useful for ASR-only diagnostics."""

    def refine(self, text: str) -> str:
        return text.strip()


class MLXRefiner:
    """MLX-LM backend for the post-trained compact Refiner."""

    def __init__(self, model_path: str, *, max_tokens: int = 512) -> None:
        try:
            from mlx_lm import load
        except ImportError as error:
            raise RuntimeError(
                "MLX inference requires `pip install mlx mlx-lm` on Apple Silicon"
            ) from error
        self.model, self.tokenizer = load(model_path)
        self.max_tokens = max_tokens

    def refine(self, text: str) -> str:
        response = self._generate(SYSTEM_PROMPT, text)
        return response.strip()

    def _generate(self, system_prompt: str, user_text: str) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        tokenizer = _chat_template_tokenizer(self.tokenizer)
        if tokenizer is None:
            raise ValueError("The Refiner tokenizer does not provide a chat template")
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=make_sampler(temp=0.0),
        )
        if response.startswith(prompt):
            response = response[len(prompt) :]
        return _clean_response(response)


def _chat_template_tokenizer(tokenizer: object) -> object | None:
    candidates = [tokenizer]
    for name in ("tokenizer", "_tokenizer", "hf_tokenizer", "processor"):
        candidate = getattr(tokenizer, name, None)
        if candidate is not None:
            candidates.append(candidate)
    return next(
        (candidate for candidate in candidates if hasattr(candidate, "apply_chat_template")),
        None,
    )


def _join_chunks(chunks: Sequence[str]) -> str:
    output = ""
    for chunk in chunks:
        value = chunk.strip()
        if not value:
            continue
        if output and output[-1].isascii() and value[0].isascii():
            output += " "
        output += value
    return output


def _clean_response(response: str) -> str:
    value = response.strip()
    value = re.sub(r"^<\|im_start\|>assistant\s*", "", value)
    value = re.sub(r"<\|im_end\|>\s*$", "", value)
    value = re.sub(r"^assistant\s*[:\uff1a]?\s*", "", value, flags=re.IGNORECASE)
    return re.sub(r"</s>\s*$", "", value).strip()
