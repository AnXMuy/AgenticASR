from __future__ import annotations

import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Final

from benchmark_io import JsonObject, Question, RubricSample

SYSTEM_PROMPT: Final = """
You are an exacting Judge for a high-semantic-noise automatic speech recognition benchmark.
You will receive the original spoken transcript (oral), the human reference transcript (clean),
the actual ASR model transcript (output), and multiple-choice questions. Judge only the output,
using oral and clean as references. For every question, select exactly one provided option.
Do not repair the output, infer words that are absent, or create scores. Return strict JSON only:
{"answers":[{"id":"q001","option_index":0}]}. option_index is zero-based. Include every question
exactly once and return no explanation or Markdown.
""".strip()
DEFAULT_API_URL: Final = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL: Final = "google/gemma-4-31b-it"


@dataclass(frozen=True, slots=True)
class ClientConfig:
    api_url: str = DEFAULT_API_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""
    timeout_seconds: float = 120.0
    max_attempts: int = 3
    temperature: float = 0.0
    system_prompt: str = SYSTEM_PROMPT


@dataclass(frozen=True, slots=True)
class JudgeAnswer:
    question_id: str
    option_index: int


class JudgeError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class JudgeResponseError(JudgeError):
    pass


def _extract_json(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end <= start:
        raise JudgeResponseError("Judge response does not contain a JSON object")
    return value[start : end + 1]


def parse_judge_response(
    text: str, option_counts: dict[str, int]
) -> tuple[JudgeAnswer, ...]:
    try:
        body = json.loads(_extract_json(text))
    except json.JSONDecodeError as error:
        raise JudgeResponseError(f"Judge returned invalid JSON: {error}") from error
    if not isinstance(body, dict) or not isinstance(body.get("answers"), list):
        raise JudgeResponseError("Judge response must contain an answers array")
    answers: list[JudgeAnswer] = []
    seen: set[str] = set()
    for raw_answer in body["answers"]:
        if not isinstance(raw_answer, dict):
            raise JudgeResponseError("Every judge answer must be an object")
        question_id = raw_answer.get("id")
        option_index = raw_answer.get("option_index")
        if not isinstance(question_id, str) or question_id not in option_counts:
            raise JudgeResponseError(f"Unknown question id: {question_id}")
        if question_id in seen:
            raise JudgeResponseError(f"Duplicate answer for {question_id}")
        if not isinstance(option_index, int) or isinstance(option_index, bool):
            raise JudgeResponseError(f"Invalid option_index for {question_id}")
        if not 0 <= option_index < option_counts[question_id]:
            raise JudgeResponseError(f"option_index out of range for {question_id}")
        seen.add(question_id)
        answers.append(JudgeAnswer(question_id, option_index))
    missing = set(option_counts) - seen
    if missing:
        raise JudgeResponseError(f"Missing answers: {', '.join(sorted(missing))}")
    return tuple(answers)


def _question_payload(question: Question) -> JsonObject:
    return {
        "id": question.question_id,
        "category": question.category,
        "question": question.text,
        "evidence": question.evidence,
        "options": [
            {"option_index": index, "text": option.text}
            for index, option in enumerate(question.options)
        ],
    }


def build_user_prompt(sample: RubricSample, output: str) -> str:
    payload: JsonObject = {
        "oral": sample.oral,
        "clean": sample.clean,
        "output": output,
        "qa": [_question_payload(question) for question in sample.questions],
    }
    return json.dumps(payload, ensure_ascii=False)


def normalize_chat_completions_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if not url:
        raise JudgeError("api_url must not be empty")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _extract_content(body: JsonObject) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise JudgeResponseError("API response has no choices")
    first = choices[0]
    if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
        raise JudgeResponseError("API response has no message")
    content = first["message"].get("content")
    if not isinstance(content, str):
        raise JudgeResponseError("API response message has no text content")
    return content


class GemmaJudgeClient:
    def __init__(self, config: ClientConfig) -> None:
        if config.max_attempts < 1:
            raise JudgeError("max_attempts must be at least 1")
        self._config = config
        self._api_url = normalize_chat_completions_url(config.api_url)

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def evaluation_identity(self) -> str:
        if (
            self._api_url == DEFAULT_API_URL
            and self._config.system_prompt == SYSTEM_PROMPT
            and self._config.temperature == 0.0
        ):
            return self._config.model
        value = {
            "api_url": self._api_url,
            "model": self._config.model,
            "system_prompt": self._config.system_prompt,
            "temperature": self._config.temperature,
        }
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:16]
        return f"{self._config.model}@{digest}"

    def judge(self, sample: RubricSample, output: str) -> tuple[JudgeAnswer, ...]:
        option_counts = {
            question.question_id: len(question.options) for question in sample.questions
        }
        last_error: JudgeError | None = None
        for attempt in range(1, self._config.max_attempts + 1):
            try:
                content = self._request(sample, output)
                return parse_judge_response(content, option_counts)
            except JudgeError as error:
                last_error = error
                if attempt < self._config.max_attempts:
                    time.sleep(2 ** (attempt - 1))
        if last_error is None:
            raise JudgeError("Judge failed without an error")
        raise last_error

    def _request(self, sample: RubricSample, output: str) -> str:
        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": self._config.system_prompt},
                {"role": "user", "content": build_user_prompt(sample, output)},
            ],
            "temperature": self._config.temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        request = urllib.request.Request(
            self._api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._config.timeout_seconds
            ) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if self._config.api_key:
                detail = detail.replace(self._config.api_key, "[REDACTED]")
            raise JudgeError(f"Judge API HTTP {error.code}: {detail[:500]}") from error
        except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
            raise JudgeError(f"Judge API connection failed: {error}") from error
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as error:
            raise JudgeResponseError(f"Judge API returned invalid JSON: {error}") from error
        if not isinstance(body, dict):
            raise JudgeResponseError("Judge API response must be a JSON object")
        return _extract_content(body)
