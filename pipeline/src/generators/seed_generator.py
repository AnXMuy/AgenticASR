# -*- coding: utf-8 -*-
"""Incremental asynchronous generation of diverse scene seed pools."""
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import (
    LLM_MAX_CONCURRENT,
    LLM_MAX_TOKENS_SEED,
    SEEDS_DIR,
    TEMPERATURE_SEED,
)
from src.services.llm_client import AsyncLLMClient, LLMClient
from src.utils.io_utils import read_json, write_json
from src.utils.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


def _normalize_seed_text(value: Any) -> str:
    """Normalize seed text for stable deduplication."""
    return " ".join(str(value).strip().lower().split())


def _seed_dedupe_key(item: Any) -> str:
    """Build a deduplication key for string or structured seeds."""
    if isinstance(item, dict):
        # Prefer the semantic value over auxiliary structured fields.
        for field in ("name", "keyword", "term", "text", "value", "title", "entity", "phrase"):
            value = item.get(field)
            if value:
                return _normalize_seed_text(value)
        return _normalize_seed_text(json.dumps(item, ensure_ascii=False, sort_keys=True))
    return _normalize_seed_text(item)


def dedupe_seed_dict(seeds: dict[str, Any]) -> dict[str, Any]:
    """Deduplicate every seed category while preserving order."""
    if not isinstance(seeds, dict):
        return seeds

    deduped: dict[str, Any] = {}
    for category, items in seeds.items():
        if not isinstance(items, list):
            deduped[category] = items
            continue

        seen: set[str] = set()
        unique_items = []
        for item in items:
            key = _seed_dedupe_key(item)
            if not key or key in seen:
                continue
            unique_items.append(item)
            seen.add(key)
        deduped[category] = unique_items

    return deduped


class SeedGenerator:
    """Generate resumable seed pools for configured scenes."""

    def __init__(self, async_client: AsyncLLMClient | None = None):
        self.llm = async_client or AsyncLLMClient(
            max_concurrent=LLM_MAX_CONCURRENT
        )
        self.loader = PromptLoader()
        self.seeds_dir = Path(SEEDS_DIR)
        self.seeds_dir.mkdir(parents=True, exist_ok=True)

    async def generate_incremental(
        self, scene: str, round_index: int
    ) -> dict[str, Any]:
        """Generate one round while excluding a summary of retained seeds."""
        existing = self.load_seeds(scene)
        exclusion_hint = self._build_exclusion_hint(existing)

        prompt = self.loader.get_seed_prompt(scene)
        user_prompt = prompt["user"]
        if exclusion_hint:
            user_prompt += f"\n\nDo not repeat these retained seeds:\n{exclusion_hint}"

        logger.info("Generating seeds for '%s' (round %d)...", scene, round_index)

        raw_output = await self.llm.generate(
            system_prompt=prompt["system"],
            user_prompt=user_prompt,
            temperature=TEMPERATURE_SEED,
            max_tokens=LLM_MAX_TOKENS_SEED,
            response_format="json_object",
        )

        new_seeds = dedupe_seed_dict(LLMClient._parse_json(raw_output))
        self.merge_seeds(scene, new_seeds)
        logger.info("Seeds for '%s' round %d merged.", scene, round_index)
        return new_seeds

    async def generate_scene_seeds(
        self, scene: str, force: bool = False
    ) -> dict[str, Any]:
        """Generate or load one scene's seed pool."""
        seed_path = self.seeds_dir / f"{scene}.json"

        if seed_path.exists() and not force:
            logger.info("Seeds exist for '%s', loading from disk.", scene)
            return read_json(seed_path)

        return await self.generate_incremental(scene, round_index=0)

    async def generate_all(self, force: bool = False) -> dict[str, dict]:
        """Generate all configured scene pools concurrently."""
        scenes = self.loader.list_scenes("seed")

        async def _gen_one(scene: str):
            return await self.generate_scene_seeds(scene, force=force)

        tasks = [_gen_one(scene) for scene in scenes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_seeds = {}
        for scene, result in zip(scenes, results):
            if isinstance(result, Exception):
                logger.error("  [FAIL] %s: %s", scene, result)
                all_seeds[scene] = {}
            else:
                all_seeds[scene] = result
        return all_seeds

    def load_seeds(self, scene: str) -> dict[str, Any]:
        """Load an existing seed pool or return an empty mapping."""
        seed_path = self.seeds_dir / f"{scene}.json"
        if seed_path.exists():
            seeds = dedupe_seed_dict(read_json(seed_path))
            write_json(seeds, seed_path)
            return seeds
        return {}

    def merge_seeds(self, scene: str, new_seeds: dict[str, Any]):
        """Merge and deduplicate an incremental seed response."""
        existing = self.load_seeds(scene)
        new_seeds = dedupe_seed_dict(new_seeds)
        if not existing:
            write_json(new_seeds, self.seeds_dir / f"{scene}.json")
            return

        for category, items in new_seeds.items():
            if category not in existing:
                existing[category] = items
            elif isinstance(items, list) and isinstance(existing[category], list):
                existing_keys = {_seed_dedupe_key(x) for x in existing[category]}
                for item in items:
                    key = _seed_dedupe_key(item)
                    if key and key not in existing_keys:
                        existing[category].append(item)
                        existing_keys.add(key)

        existing = dedupe_seed_dict(existing)
        write_json(existing, self.seeds_dir / f"{scene}.json")
        logger.info("Merged seeds for '%s'", scene)

    def _build_exclusion_hint(self, seeds: dict[str, Any], max_items: int = 30) -> str:
        """Build a compact exclusion summary from retained seeds."""
        if not seeds:
            return ""
        hints = []
        for category, items in seeds.items():
            if isinstance(items, list) and items:
                sample = items[:min(5, len(items))]
                names = []
                for s in sample:
                    if isinstance(s, dict):
                        names.append(s.get("name", s.get("artist", str(s)[:20])))
                    else:
                        names.append(str(s)[:20])
                hints.append(f"{category}: {', '.join(names)}...")
            if len(hints) >= max_items:
                break
        return "\n".join(hints)
