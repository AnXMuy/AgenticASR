# -*- coding: utf-8 -*-
"""Orchestrate the complete AgenticASR training-data pipeline."""
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))

from configs.scenes import SCENE_REGISTRY
from configs.settings import ASR_MODE, VLLM_BASE_URL
from src.services.vllm_lifecycle import (
    ensure_speech_services_ready,
    ensure_vllm_started,
    ensure_vllm_stopped,
)

logger = logging.getLogger("run_pipeline")

PYTHON = sys.executable


def run_cmd(cmd: list[str], step_name: str) -> bool:
    logger.info("[START] %s", step_name)
    logger.info("  CMD: %s", " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        logger.error("[FAIL] %s (%.1fs, exit=%d)", step_name, elapsed, result.returncode)
        return False
    logger.info("[DONE] %s (%.1fs)", step_name, elapsed)
    return True


def step1_seed_oral(args):
    """Run Seed and Oral generation."""
    cmd = [PYTHON, str(SCRIPTS_DIR / "01_seed_and_oral.py")]
    if args.scene:
        cmd += ["--scene", args.scene]
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    if args.max_concurrent:
        cmd += ["--max-concurrent", str(args.max_concurrent)]
    if args.force_seeds:
        cmd += ["--force-seeds"]
    return run_cmd(cmd, "Step 1: Seed + Oral Text")


def step2_clean_text(args):
    """Run Oral-to-Clean generation."""
    cmd = [PYTHON, str(SCRIPTS_DIR / "02_clean_text.py")]
    if args.scene:
        cmd += ["--scene", args.scene]
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    return run_cmd(cmd, "Step 2: Clean Text (Oral -> Clean)")


def step3_asr_sim(args):
    """Run aligned ASR simulation."""
    cmd = [PYTHON, str(SCRIPTS_DIR / "03_simulate_asr.py")]
    if args.scene:
        cmd += ["--scene", args.scene]
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    if args.asr_mode:
        cmd += ["--asr-mode", args.asr_mode]
    if args.asr_strength:
        cmd += ["--asr-strength", args.asr_strength]
    if args.tts_provider:
        cmd += ["--tts-provider", args.tts_provider]
    if args.stt_provider:
        cmd += ["--stt-provider", args.stt_provider]
    if args.tts_base_url:
        cmd += ["--tts-base-url", args.tts_base_url]
    if args.stt_base_url:
        cmd += ["--stt-base-url", args.stt_base_url]
    return run_cmd(cmd, "Step 3: ASR Simulation")


def step4_assemble(args):
    """Materialize common training records."""
    cmd = [PYTHON, str(SCRIPTS_DIR / "04_assemble.py")]
    if args.scene:
        cmd += ["--scene", args.scene]
    return run_cmd(cmd, "Step 4: Assemble Dataset")


def step5_finalize(args):
    """Run quality control and deduplication."""
    cmd = [PYTHON, str(SCRIPTS_DIR / "05_finalize.py")]
    if args.base_url:
        cmd += ["--base-url", args.base_url]
    return run_cmd(cmd, "Step 5: Finalize Dataset")


def get_scenes(args) -> list[str]:
    if args.scene:
        return [args.scene]
    return list(SCENE_REGISTRY.keys())


def _llm_base_url(args) -> str:
    return args.base_url or VLLM_BASE_URL


def _asr_mode(args) -> str:
    return args.asr_mode or ASR_MODE


def prepare_step(step: int, args) -> None:
    if step in {1, 2, 5}:
        ensure_vllm_started(_llm_base_url(args))
    elif step == 3:
        mode = _asr_mode(args)
        if mode == "llm_sim":
            ensure_vllm_started(_llm_base_url(args))
        else:
            ensure_vllm_stopped()
        if mode == "tts_asr":
            ensure_speech_services_ready(args.tts_base_url, args.stt_base_url)
    elif step == 4:
        ensure_vllm_stopped()


def main():
    parser = argparse.ArgumentParser(
        description="Run the AgenticASR data-generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--step", type=int, default=1,
                        help="first step to run (1-5, default: 1)")
    parser.add_argument("--only", type=int, default=None,
                        help="run only one step")
    parser.add_argument("--scene", type=str, default=None,
                        help="process one internal scene key")
    parser.add_argument("--base-url", type=str, default=None,
                        help="LLM service URL")
    parser.add_argument("--max-concurrent", type=int, default=16,
                        help="maximum LLM concurrency")
    parser.add_argument("--force-seeds", action="store_true",
                        help="regenerate existing seed pools")
    parser.add_argument("--asr-mode", type=str, default=None, choices=["llm_sim", "text_sim", "tts_asr"],
                        help="Step 3 input-generation mode")
    parser.add_argument("--asr-strength", type=str, default=None, choices=["light", "medium", "heavy"],
                        help="diagnostic text-simulation strength")
    parser.add_argument("--tts-provider", type=str, default=None, choices=["moss_api"],
                        help="TTS provider for tts_asr mode")
    parser.add_argument("--stt-provider", type=str, default=None, choices=["qwen3_api"],
                        help="STT provider for tts_asr mode")
    parser.add_argument("--tts-base-url", type=str, default=None,
                        help="MOSS-TTS service URL")
    parser.add_argument("--stt-base-url", type=str, default=None,
                        help="Qwen3-ASR service URL")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    scenes = get_scenes(args)
    start = args.only if args.only else args.step
    end = args.only if args.only else 5

    logger.info("=" * 50)
    logger.info("  AgenticASR Pipeline")
    logger.info("  Steps: %d -> %d | Scenes: %s", start, end, ", ".join(scenes))
    logger.info("=" * 50)

    t_total = time.time()

    if start <= 1 <= end:
        prepare_step(1, args)
        if not step1_seed_oral(args):
            sys.exit(1)

    if start <= 2 <= end:
        prepare_step(2, args)
        if not step2_clean_text(args):
            sys.exit(1)

    if start <= 3 <= end:
        prepare_step(3, args)
        if not step3_asr_sim(args):
            sys.exit(1)

    if start <= 4 <= end:
        prepare_step(4, args)
        if not step4_assemble(args):
            sys.exit(1)

    if start <= 5 <= end:
        prepare_step(5, args)
        if not step5_finalize(args):
            sys.exit(1)

    elapsed_total = time.time() - t_total
    logger.info("=" * 50)
    logger.info("  Pipeline DONE! Total: %.1fs (%.1f min)", elapsed_total, elapsed_total / 60)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
