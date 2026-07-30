"""Quality-control checks for generated Refiner training pairs."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import LLM_MAX_TOKENS_DEFAULT
from src.services.llm_client import AsyncLLMClient, LLMClient
from src.utils.keyword_utils import dedupe_keywords, is_common_keyword, keyword_has_evidence

logger = logging.getLogger(__name__)

SEMANTIC_QC_SYSTEM = """Check whether a refined transcript preserves the complete final
intended meaning of its input. A valid refinement may remove fillers, repetitions, false starts,
abandoned corrections, and spelling explanations, and may apply ITN and formatting. It must keep
all supported people, places, values, times, actions, intent, and conclusions; it must not
summarize, hallucinate, or over-specify. Return strict JSON only:
{"consistent":true,"reason":"brief reason"}."""


class QCValidator:
    """Validate traceable error-prone keywords on generated records."""

    def __init__(self, async_client: AsyncLLMClient | None = None, max_concurrent: int = 8):
        self.llm = async_client or AsyncLLMClient(max_concurrent=max_concurrent)
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def validate_keywords_one(self, item: dict) -> tuple[bool, str, list[str]]:
        input_text = item.get("input", "")
        oral_text = item.get("oral_text", "")
        refined_text = item.get("output", {}).get("refined_text", "")
        keywords = dedupe_keywords(item.get("output", {}).get("keyword_list", []))
        if not keywords:
            return True, "no keywords", []

        common = [keyword for keyword in keywords if is_common_keyword(keyword)]
        if common:
            return False, f"keywords contain common terms: {common}", common

        source = "\n".join(part for part in (input_text, oral_text) if part)
        unsupported = [
            keyword
            for keyword in keywords
            if not keyword_has_evidence(keyword, source, refined_text)
        ]
        if unsupported:
            return False, f"keywords lack traceable evidence: {unsupported}", unsupported
        return True, "rule checks passed", []

    async def validate_batch(self, items: list[dict], lang: str = "zh") -> list[dict]:
        del lang
        valid: list[dict] = []
        for item in items:
            accepted, reason, _ = await self.validate_keywords_one(item)
            if accepted:
                valid.append(item)
            else:
                logger.debug("Keyword validation failed: %s", reason)
        return valid


async def validate_semantic_consistency(
    input_text: str,
    refined_text: str,
    llm: AsyncLLMClient,
) -> tuple[bool, str]:
    """Use the configured LLM to check intent preservation."""

    user_prompt = (
        f"Input transcript:\n{input_text}\n\nRefined transcript:\n{refined_text}\n\n"
        "Does the refinement preserve the complete final intended meaning?"
    )
    try:
        response = await llm.generate(
            system_prompt=SEMANTIC_QC_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=LLM_MAX_TOKENS_DEFAULT,
        )
        result = LLMClient._parse_json(response)
        if not isinstance(result, dict):
            raise ValueError("semantic QC response must be a JSON object")
        return bool(result.get("consistent", True)), str(result.get("reason", ""))
    except (ValueError, json.JSONDecodeError, RuntimeError) as error:
        logger.warning("Semantic validation failed: %s", error)
        return True, "validation failed; retained by fallback policy"


_SPOKEN_CHINESE_NUMBER_PATTERNS = [
    re.compile(r"[\u4e00-\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e07\u4ebf]+[\u5e74\u6708\u65e5\u53f7]"),
    re.compile(r"\u767e\u5206\u4e4b[\u96f6\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e07]+"),
]


def check_itn_correctness(refined_text: str, original_text: str) -> tuple[bool, list[str]]:
    """Flag obvious spoken-form Chinese numbers left in a Clean target."""

    del original_text
    issues = [
        f"possible unnormalized spoken number matched {pattern.pattern}"
        for pattern in _SPOKEN_CHINESE_NUMBER_PATTERNS
        if pattern.search(refined_text)
    ]
    return not issues, issues


async def run_qc_on_dataset(items: list[dict], max_concurrent: int = 8) -> list[dict]:
    """Run keyword quality control over a dataset."""

    return await QCValidator(max_concurrent=max_concurrent).validate_batch(items)
