# -*- coding: utf-8 -*-
"""Synchronous and asynchronous clients for vLLM or OpenRouter."""
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAI

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import (
    LLM_BASE_URL,
    LLM_MODEL_NAME,
    LLM_API_KEY,
    USE_OPENROUTER,
    LLM_MAX_TOKENS_DEFAULT,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Synchronous OpenAI-compatible LLM client with retries."""

    def __init__(
        self,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL_NAME,
        api_key: str = LLM_API_KEY,
        max_retries: int = 3,
        timeout: float = 120.0,
        use_reasoning: bool = False,
    ):
        self.model = model
        self.max_retries = max_retries
        self.use_reasoning = use_reasoning
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        logger.info("LLMClient initialized: base_url=%s, model=%s", base_url, model)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = LLM_MAX_TOKENS_DEFAULT,
        response_format: str | None = None,
    ) -> str:
        """Generate one response with retry handling."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # OpenRouter supports an optional reasoning request body.
        if USE_OPENROUTER and self.use_reasoning:
            kwargs["extra_body"] = {"reasoning": {"enabled": True}}
        elif response_format == "json_object":
            kwargs["extra_body"] = {"guided_json": None}
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.time()
                response = self.client.chat.completions.create(**kwargs)
                elapsed = time.time() - t0
                content = response.choices[0].message.content or ""
                logger.debug(
                    "LLM generate OK: %d tokens in %.1fs",
                    response.usage.completion_tokens if response.usage else 0,
                    elapsed,
                )
                return content
            except Exception as e:
                logger.warning(
                    "LLM generate attempt %d/%d failed: %s",
                    attempt, self.max_retries, e,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)

        return ""  # unreachable, but keeps type checker happy

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = LLM_MAX_TOKENS_DEFAULT,
    ) -> Any:
        """Generate and parse a JSON response."""
        raw = self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json_object",
        )
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Extract JSON from common plain-text or fenced model responses."""
        text = text.strip()

        # Remove an optional Markdown code fence.
        if text.startswith("```"):
            if "\n" in text:
                end_of_first = text.index("\n")
                text = text[end_of_first + 1:]
            else:
                text = text[3:]
            # Remove the closing fence and trailing whitespace.
            last_fence = text.rfind("```")
            if last_fence != -1:
                text = text[:last_fence]
            text = text.strip()

        # Prefer direct parsing.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try every plausible JSON start.
        best_error = None
        for i, ch in enumerate(text):
            if ch in ("[", "{"):
                try:
                    return json.loads(text[i:])
                except json.JSONDecodeError as e:
                    if best_error is None:
                        best_error = e
                    continue

        # Finally, scan backward for the last complete JSON structure.
        for end_char, start_char in (("]", "["), ("}", "{")):
            last_end = text.rfind(end_char)
            if last_end != -1:
                # Find the matching opening delimiter.
                depth = 0
                start = -1
                for i in range(last_end, -1, -1):
                    if text[i] == end_char:
                        depth += 1
                    elif text[i] == start_char:
                        depth -= 1
                        if depth == 0:
                            start = i
                            break
                if start != -1:
                    try:
                        return json.loads(text[start:last_end + 1])
                    except json.JSONDecodeError:
                        pass

        logger.error("Failed to parse JSON from LLM output: %s...", text[:200])
        raise ValueError(f"Cannot parse JSON from LLM output: {text[:200]}")


class AsyncLLMClient:
    """Asynchronous OpenAI-compatible LLM client with concurrency control."""

    def __init__(
        self,
        base_url: str = LLM_BASE_URL,
        model: str = LLM_MODEL_NAME,
        api_key: str = LLM_API_KEY,
        max_retries: int = 3,
        timeout: float = 120.0,
        max_concurrent: int = 8,
        use_reasoning: bool = False,
    ):
        self.model = model
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.use_reasoning = use_reasoning
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self._semaphore: asyncio.Semaphore | None = None
        logger.info("AsyncLLMClient initialized: base_url=%s, model=%s, max_concurrent=%d",
                    base_url, model, max_concurrent)

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = LLM_MAX_TOKENS_DEFAULT,
        response_format: str | None = None,
    ) -> str:
        """Generate one response asynchronously."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # OpenRouter supports an optional reasoning request body.
        if USE_OPENROUTER and self.use_reasoning:
            kwargs["extra_body"] = {"reasoning": {"enabled": True}}
        elif response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        async with self.semaphore:
            for attempt in range(1, self.max_retries + 1):
                try:
                    t0 = time.time()
                    response = await self.client.chat.completions.create(**kwargs)
                    elapsed = time.time() - t0
                    content = response.choices[0].message.content or ""
                    logger.debug(
                        "Async LLM generate OK: %d tokens in %.1fs",
                        response.usage.completion_tokens if response.usage else 0,
                        elapsed,
                    )
                    return content
                except Exception as e:
                    logger.warning(
                        "Async LLM attempt %d/%d failed: %s",
                        attempt, self.max_retries, e,
                    )
                    if attempt == self.max_retries:
                        raise
                    await asyncio.sleep(2 ** attempt)

        return ""

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = LLM_MAX_TOKENS_DEFAULT,
    ) -> Any:
        """Generate and parse one JSON response asynchronously."""
        raw = await self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json_object",
        )
        return LLMClient._parse_json(raw)

    async def batch_generate(
        self,
        requests: list[dict[str, Any]],
    ) -> list[str]:
        """Generate a batch of responses concurrently."""
        tasks = [self.generate(**req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def batch_generate_json(
        self,
        requests: list[dict[str, Any]],
    ) -> list[Any]:
        """Generate and parse a batch of JSON responses concurrently."""
        tasks = [self.generate_json(**req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)


# Convenience constructors.

def create_client(**kwargs) -> LLMClient:
    """Create a synchronous client."""
    return LLMClient(**kwargs)


def create_async_client(**kwargs) -> AsyncLLMClient:
    """Create an asynchronous client."""
    return AsyncLLMClient(**kwargs)


def get_llm_info() -> dict[str, str]:
    """Return a credential-safe summary of the current configuration."""
    return {
        "use_openrouter": str(USE_OPENROUTER),
        "base_url": LLM_BASE_URL,
        "model": LLM_MODEL_NAME,
        "api_key_set": "yes" if LLM_API_KEY and LLM_API_KEY != "EMPTY" else "no",
    }


if __name__ == "__main__":
    # Print configuration without exposing credentials.
    logging.basicConfig(level=logging.DEBUG)

    info = get_llm_info()
    print("=== LLM Configuration ===")
    for key, value in info.items():
        print(f"  {key}: {value}")
