# -*- coding: utf-8 -*-
"""Step 1: generate scene seed pools and Oral utterances."""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import (
    RAW_DIR,
    ROUND_SIZE,
    SCENE_DISTRIBUTION,
    SEEDS_DIR,
    SEED_TARGET_COUNT,
    SEED_PER_ROUND,
    TOTAL_SAMPLES,
)
from src.generators.oral_text_generator import OralTextGenerator
from src.generators.seed_generator import SeedGenerator, dedupe_seed_dict
from src.services.llm_client import AsyncLLMClient
from src.utils.io_utils import read_jsonl, setup_logging, write_json
from src.utils.prompt_loader import PromptLoader


def _existing_oral_count(scene: str, resume: bool) -> int:
    scene_dir = Path(RAW_DIR) / scene
    output_path = scene_dir / "oral.jsonl"
    if not resume or not output_path.exists():
        return 0
    return len(read_jsonl(output_path))


def _oral_generation_target(
    scene: str,
    distribution_target: int,
    rounds: int | None,
    resume: bool,
) -> tuple[int, int, int]:
    """Return generated count, target record count, and planned incremental rounds."""
    generated = _existing_oral_count(scene, resume=resume)
    if rounds is not None:
        return generated, generated + rounds * ROUND_SIZE, rounds

    generation_target = max(distribution_target, generated)
    remaining = max(0, generation_target - generated)
    planned_rounds = (remaining + ROUND_SIZE - 1) // ROUND_SIZE
    return generated, generation_target, planned_rounds


# ============================================================
# Phase 1: concurrent seed generation.
# ============================================================

def _calc_seed_rounds(scene: str) -> int:
    """Compute the rounds needed to reach the seed target."""
    seeds_dir = Path(SEEDS_DIR)
    seed_path = seeds_dir / f"{scene}.json"

    # Count existing unique seeds.
    existing_count = 0
    if seed_path.exists():
        import json
        with open(seed_path, "r", encoding="utf-8") as f:
            existing = dedupe_seed_dict(json.load(f))
        write_json(existing, seed_path)
        for category, items in existing.items():
            if isinstance(items, list):
                existing_count += len(items)

    # Compute the remaining rounds.
    remaining = max(0, SEED_TARGET_COUNT - existing_count)
    if remaining == 0:
        return 0

    rounds = (remaining + SEED_PER_ROUND - 1) // SEED_PER_ROUND
    return max(1, rounds)


async def phase1_seeds(
    scenes: list[str],
    base_url: str | None,
    max_concurrent: int,
    logger: logging.Logger,
):
    """Generate missing seed rounds for every scene concurrently."""
    logger.info("=== Phase 1: Seed Generation (%d scenes, parallel) ===", len(scenes))
    logger.info("  SEED_TARGET_COUNT: %d, SEED_PER_ROUND: %d", SEED_TARGET_COUNT, SEED_PER_ROUND)

    client_kwargs = {"max_concurrent": max_concurrent}
    if base_url:
        client_kwargs["base_url"] = base_url

    async def _seed_one(scene: str):
        rounds_needed = _calc_seed_rounds(scene)
        if rounds_needed == 0:
            logger.info("[Seed] %s: already have enough seeds, skip", scene)
            return

        llm = AsyncLLMClient(**client_kwargs)
        gen = SeedGenerator(async_client=llm)
        logger.info("[Seed] %s: %d round(s) needed", scene, rounds_needed)

        for r in range(rounds_needed):
            await gen.generate_incremental(scene, round_index=r)
            logger.info("[Seed] %s: round %d/%d done", scene, r + 1, rounds_needed)

    results = await asyncio.gather(*[_seed_one(s) for s in scenes], return_exceptions=True)
    for scene, r in zip(scenes, results):
        if isinstance(r, Exception):
            logger.error("[Seed] %s: FAILED: %s", scene, r)
    logger.info("=== Phase 1 done ===")


# ============================================================
# Phase 2: sequential scenes with concurrent requests per scene.
# ============================================================

async def phase2_oral(
    scenes: list[str],
    targets: dict[str, int],
    base_url: str | None,
    resume: bool,
    logger: logging.Logger,
):
    logger.info("=== Phase 2: Oral Text (%d scenes) ===", len(scenes))
    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url

    for scene in scenes:
        target = targets[scene]
        scene_dir = Path(RAW_DIR) / scene
        scene_dir.mkdir(parents=True, exist_ok=True)

        llm = AsyncLLMClient(**client_kwargs)
        oral_gen = OralTextGenerator(async_client=llm)

        generated = await oral_gen.generate_scene(
            scene=scene,
            total_num=target,
            resume=resume,
        )
        logger.info("[Oral] %s: done, %d/%d items", scene, generated, target)

    logger.info("=== Phase 2 done ===")


# ============================================================
# CLI entry point.
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Seed pools and Oral text")
    parser.add_argument("--scene", type=str, default=None, help="one internal scene key")
    parser.add_argument("--rounds", type=int, default=None, help="override Oral round count")
    parser.add_argument("--total", type=int, default=TOTAL_SAMPLES, help="target record count")
    parser.add_argument("--base-url", type=str, default=None, help="vLLM service URL")
    parser.add_argument("--max-concurrent", type=int, default=16, help="per-scene concurrency")
    parser.add_argument("--force-seeds", action="store_true", help="regenerate seed pools")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.rounds is not None and args.rounds < 1:
        parser.error("--rounds must be at least 1")

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    loader = PromptLoader()
    available = loader.list_scenes("seed")
    if args.scene:
        scenes = [args.scene]
    else:
        scenes = [s for s in available if s != "passthrough"]

    logger.info("=== Step 1: Seed + Oral Text ===")
    logger.info("Scenes (%d): %s", len(scenes), ", ".join(scenes))

    targets: dict[str, int] = {}
    plan_map: dict[str, tuple[int, int, int, int]] = {}
    for scene in scenes:
        distribution_target = int(args.total * SCENE_DISTRIBUTION.get(scene, 0.05))
        generated, generation_target, planned_rounds = _oral_generation_target(
            scene,
            distribution_target,
            args.rounds,
            resume=not args.no_resume,
        )
        targets[scene] = generation_target
        plan_map[scene] = (
            distribution_target,
            generated,
            generation_target,
            planned_rounds,
        )

    for scene in scenes:
        distribution_target, generated, generation_target, planned_rounds = plan_map[scene]
        seed_rounds = _calc_seed_rounds(scene)
        logger.info(
            "  %s: distribution_target=%d, oral_records=%d->%d, "
            "oral_rounds=%d, seed_rounds=%d",
            scene,
            distribution_target,
            generated,
            generation_target,
            planned_rounds,
            seed_rounds,
        )

    # Skip seed generation when all scene pools already exist.
    seeds_dir = Path(SEEDS_DIR)
    need_seed_generation = args.force_seeds
    if not args.force_seeds:
        for scene in scenes:
            seed_path = seeds_dir / f"{scene}.json"
            if not seed_path.exists():
                need_seed_generation = True
                logger.info("[Seed] %s: seed file not found, will generate", scene)
                break
        if not need_seed_generation:
            logger.info("=== Phase 1 SKIPPED (all seed files exist, use --force-seeds to regenerate) ===")

    async def run_all():
        if need_seed_generation:
            await phase1_seeds(scenes, args.base_url, args.max_concurrent, logger)
        else:
            logger.info("=== Phase 1 SKIPPED ===")
        await phase2_oral(scenes, targets, args.base_url, not args.no_resume, logger)

    asyncio.run(run_all())
    logger.info("All done.")


if __name__ == "__main__":
    main()
