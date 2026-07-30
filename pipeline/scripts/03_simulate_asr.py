# -*- coding: utf-8 -*-
"""Step 3: generate aligned ASR-style inputs from Oral/Clean records."""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import (
    ASR_DEVICE,
    ASR_FALLBACK_TEXT_SIM,
    ASR_MODE,
    ASR_STRENGTH,
    ASR_STT_BASE_URL,
    ASR_STT_PROVIDER,
    ASR_TTS_BASE_URL,
    ASR_TTS_PROVIDER,
    RAW_DIR,
)
from src.generators.asr_simulator import ASRSimulator
from src.services.llm_client import AsyncLLMClient
from src.utils.io_utils import setup_logging


async def run(args):
    logger = logging.getLogger(__name__)

    llm = None
    if args.asr_mode == "llm_sim":
        client_kwargs = {}
        if args.base_url:
            client_kwargs["base_url"] = args.base_url
        llm = AsyncLLMClient(**client_kwargs)

    sim = ASRSimulator(
        async_client=llm,
        mode=args.asr_mode,
        strength=args.asr_strength,
        tts_provider=args.tts_provider,
        stt_provider=args.stt_provider,
        tts_base_url=args.tts_base_url,
        stt_base_url=args.stt_base_url,
        device=args.device,
        enable_cache=not args.no_cache,
        fallback_text_sim=args.fallback_text_sim,
    )

    raw_dir = Path(RAW_DIR)
    source_file = "clean.jsonl" if args.input_source == "clean" else "oral.jsonl"
    available_scenes = sorted([
        d.name for d in raw_dir.iterdir()
        if d.is_dir() and (d / source_file).exists()
    ]) if raw_dir.exists() else []

    if args.scene:
        if args.scene not in available_scenes:
            logger.error("Scene '%s' not found for source %s. Available: %s",
                         args.scene, source_file, available_scenes)
            sys.exit(1)
        scenes = [args.scene]
    else:
        scenes = available_scenes

    logger.info(
        "Processing %d scene(s): %s | source=%s | mode=%s | strength=%s",
        len(scenes), ", ".join(scenes), source_file, args.asr_mode, args.asr_strength,
    )
    tasks = [
        sim.simulate_scene(scene, resume=not args.no_resume, input_source=args.input_source)
        for scene in scenes
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_processed = 0
    for scene, result in zip(scenes, results):
        if isinstance(result, Exception):
            logger.error("  [FAIL] %s: %s", scene, result)
        else:
            total_processed += result
            logger.info("  [OK] %s: %d items", scene, result)
    logger.info("All done. Total processed: %d", total_processed)


def main():
    parser = argparse.ArgumentParser(description="Generate aligned ASR-style inputs")
    parser.add_argument("--scene", type=str, default=None, help="one internal scene key")
    parser.add_argument("--no-resume", action="store_true", help="restart from the beginning")
    parser.add_argument("--base-url", type=str, default=None, help="LLM service URL")
    parser.add_argument("--asr-mode", type=str, default=ASR_MODE,
                        choices=["llm_sim", "text_sim", "tts_asr"], help="input-generation mode")
    parser.add_argument("--asr-strength", type=str, default=ASR_STRENGTH,
                        choices=["light", "medium", "heavy"], help="diagnostic simulation strength")
    parser.add_argument("--input-source", type=str, default="clean",
                        choices=["clean", "oral"], help="Step 3 source records")
    parser.add_argument("--tts-provider", type=str, default=ASR_TTS_PROVIDER,
                        choices=["moss_api"], help="TTS provider for tts_asr mode")
    parser.add_argument("--stt-provider", type=str, default=ASR_STT_PROVIDER,
                        choices=["qwen3_api"], help="STT provider for tts_asr mode")
    parser.add_argument("--tts-base-url", type=str, default=ASR_TTS_BASE_URL,
                        help="MOSS-TTS service URL")
    parser.add_argument("--stt-base-url", type=str, default=ASR_STT_BASE_URL,
                        help="Qwen3-ASR service URL")
    parser.add_argument("--device", type=str, default=ASR_DEVICE, help="tts_asr device")
    parser.add_argument("--no-cache", action="store_true", help="disable tts_asr caching")
    parser.add_argument("--fallback-text-sim", dest="fallback_text_sim", action="store_true",
                        help="fall back to text_sim after a tts_asr failure")
    parser.add_argument("--no-fallback-text-sim", dest="fallback_text_sim", action="store_false",
                        help="fail immediately after a tts_asr error")
    parser.set_defaults(fallback_text_sim=ASR_FALLBACK_TEXT_SIM)
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
