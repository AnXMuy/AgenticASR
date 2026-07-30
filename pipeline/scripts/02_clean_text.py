# -*- coding: utf-8 -*-
"""Step 2: generate Clean targets from Oral records."""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import RAW_DIR
from src.generators.clean_text_generator import CleanTextGenerator
from src.services.llm_client import AsyncLLMClient
from src.utils.io_utils import setup_logging


async def run(args):
    logger = logging.getLogger(__name__)

    client_kwargs = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    llm = AsyncLLMClient(**client_kwargs)
    gen = CleanTextGenerator(async_client=llm)

    raw_dir = Path(RAW_DIR)
    available_scenes = sorted([
        d.name for d in raw_dir.iterdir()
        if d.is_dir() and (d / "oral.jsonl").exists()
    ])

    if args.scene:
        if args.scene not in available_scenes:
            logger.error("Scene '%s' not found. Available: %s", args.scene, available_scenes)
            sys.exit(1)
        scenes = [args.scene]
    else:
        scenes = available_scenes

    logger.info("Processing %d scene(s): %s", len(scenes), ", ".join(scenes))
    total_processed = 0
    for scene in scenes:
        count = await gen.generate_scene(scene, resume=not args.no_resume)
        total_processed += count
        logger.info("  [OK] %s: %d items", scene, count)
    logger.info("All done. Total processed: %d", total_processed)


def main():
    parser = argparse.ArgumentParser(description="Generate Clean targets from Oral records")
    parser.add_argument("--scene", type=str, default=None, help="one internal scene key")
    parser.add_argument("--no-resume", action="store_true", help="restart from the beginning")
    parser.add_argument("--base-url", type=str, default=None, help="vLLM service URL")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
