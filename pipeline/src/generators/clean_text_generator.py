# -*- coding: utf-8 -*-
"""Generate Clean targets from Oral text and apply quality control."""
import asyncio
import json
import logging
import random
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import (
    LLM_MAX_CONCURRENT,
    LLM_MAX_TOKENS_CLEAN,
    RAW_DIR,
)
from src.pipeline.quality_control import QCValidator
from src.services.llm_client import AsyncLLMClient, LLMClient
from src.utils.io_utils import Checkpoint, append_jsonl, read_jsonl
from src.utils.keyword_utils import normalize_review_targets

from tqdm import tqdm

logger = logging.getLogger(__name__)

CONFIGS_DIR = PROJECT_ROOT / "configs"

_QC_ENABLED = True
_QC_CONCURRENT = 8
_TEMP_RANGE = (0.15, 0.35)


def _load_clean_prompts() -> dict:
    with open(CONFIGS_DIR / "prompts_clean.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_clean_response(raw_output: str) -> dict | None:
    """Parse a structured Clean-generation response."""
    try:
        result = LLMClient._parse_json(raw_output)
    except (ValueError, json.JSONDecodeError):
        return None

    if isinstance(result, list):
        result = result[0] if result else {}
    elif isinstance(result, dict):
        for key in ("data", "results", "sentences", "items", "instructions"):
            if key in result and isinstance(result[key], list):
                arr = result[key]
                result = arr[0] if arr else {}
                break

    if not isinstance(result, dict):
        return None

    text = result.get("clean_text", "")
    if not text or len(text.strip()) < 4:
        return None

    return {
        "clean_text": text.strip(),
        "domain_keywords": result.get("domain_keywords", []),
        "itn_types": result.get("itn_types", []),
    }


class CleanTextGenerator:
    """Generate Clean targets asynchronously and validate keywords."""

    def __init__(self, async_client: AsyncLLMClient | None = None):
        self.llm = async_client or AsyncLLMClient(max_concurrent=LLM_MAX_CONCURRENT)
        self.prompts = _load_clean_prompts()
        self.raw_dir = Path(RAW_DIR)
        self.qc = QCValidator() if _QC_ENABLED else None

    def _get_lang(self, scene: str) -> str:
        return "en" if scene.startswith("english_") else "zh"

    def _get_system_prompt(self, scene: str) -> str:
        prompts = self.prompts["system_prompts"]
        return prompts.get(scene, prompts.get("zh", ""))

    def _sample_temp(self) -> float:
        """Sample a low generation temperature."""
        return random.uniform(*_TEMP_RANGE)

    async def generate_one(self, scene: str, item: dict) -> dict | None:
        """Generate one Clean target."""
        oral_text = item.get("oral_text", "")
        if not oral_text or len(oral_text) < 4:
            return None

        phenomena = item.get("speech_phenomena") or []
        if any(str(p).lower() in {"clean", "passthrough"} for p in phenomena):
            return {
                **item,
                "clean_text": oral_text,
                "domain_keywords": [],
                "keyword_list": [],
                "itn_types": [],
            }

        lang = self._get_lang(scene)
        system = self._get_system_prompt(lang)
        template = self.prompts["user_template"][lang]
        user = template.format(oral_text=oral_text)

        try:
            raw_output = await self.llm.generate(
                system_prompt=system,
                user_prompt=user,
                temperature=self._sample_temp(),
                max_tokens=LLM_MAX_TOKENS_CLEAN,
                response_format="json_object",
            )
            parsed = _parse_clean_response(raw_output)
            if parsed is None:
                return None
            clean_text = parsed["clean_text"]
            keywords = normalize_review_targets(
                parsed.get("domain_keywords"),
                input_text=oral_text,
                refined_text=clean_text,
            )
            parsed["domain_keywords"] = keywords
            parsed["keyword_list"] = keywords
            return {**item, **parsed}
        except Exception as e:
            logger.warning("Clean generation failed for id=%s: %s", item.get("id"), e)
            return None

    async def generate_scene(self, scene: str, resume: bool = True) -> int:
        """Generate and validate all pending records for one scene."""
        scene_dir = self.raw_dir / scene
        oral_path = scene_dir / "oral.jsonl"
        clean_path = scene_dir / "clean.jsonl"
        ckpt_path = scene_dir / "clean.ckpt.json"

        if not oral_path.exists():
            logger.error("Oral text not found: %s", oral_path)
            return 0

        oral_data = read_jsonl(oral_path)
        total = len(oral_data)
        if not resume and clean_path.exists():
            clean_path.unlink()
        generated_count = len(read_jsonl(clean_path)) if resume and clean_path.exists() else 0

        for i, item in enumerate(oral_data):
            if "id" not in item:
                item["id"] = i + 1

        ckpt = Checkpoint(ckpt_path)
        source_offset = ckpt.get("source_offset", ckpt.get("processed_count", 0)) if resume else 0
        remaining_items = oral_data[source_offset:]

        if not remaining_items:
            logger.info("Scene '%s': all %d items already processed.", scene, total)
            return generated_count

        lang = self._get_lang(scene)
        qc_enabled = _QC_ENABLED and self.qc is not None

        logger.info("Scene '%s': total=%d, processed=%d, remaining=%d (QC=%s)",
                    scene, total, source_offset, len(remaining_items), qc_enabled)

        batch_size = LLM_MAX_CONCURRENT * 2
        pbar = tqdm(total=len(remaining_items), desc=f"Clean[{scene}]", unit="item")

        total_qc_rejected = 0

        for i in range(0, len(remaining_items), batch_size):
            batch_items = remaining_items[i:i + batch_size]
            tasks = [self.generate_one(scene, item) for item in batch_items]
            results = await asyncio.gather(*tasks)
            valid = [r for r in results if r is not None]

            # Keyword and traceability quality control.
            if qc_enabled and valid:
                before_qc = len(valid)
                valid = await self.qc.validate_batch(valid, lang)
                total_qc_rejected += before_qc - len(valid)

            if valid:
                append_jsonl(valid, clean_path)
                generated_count += len(valid)
            source_offset += len(batch_items)
            ckpt.set("source_offset", source_offset)
            ckpt.set("generated_count", generated_count)
            pbar.update(len(batch_items))

        pbar.close()

        if qc_enabled and total_qc_rejected > 0:
            logger.info("Scene '%s': QC rejected %d pairs total", scene, total_qc_rejected)

        return generated_count
