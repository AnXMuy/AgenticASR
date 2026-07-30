# -*- coding: utf-8 -*-
"""Finalize generated pairs with semantic QC, deduplication, and statistics."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import (
    DEDUP_THRESHOLD,
    FINAL_DIR,
    LLM_MAX_CONCURRENT,
    LLM_MAX_TOKENS_QC,
    PROCESSED_DIR,
    SCENE_DISTRIBUTION,
    TOTAL_SAMPLES,
)
from src.pipeline.dedup import compute_distribution, deduplicate
from src.services.llm_client import AsyncLLMClient
from src.utils.io_utils import read_jsonl, setup_logging, write_jsonl

from tqdm import tqdm

logger = logging.getLogger("finalize")

SEMANTIC_QC_SYSTEM = """Judge whether a Refiner target exactly captures the speaker's
final intended meaning. Resolve single, rollback, delayed, and multi-stage corrections using the
final value. Permit filler/repetition removal, ITN, punctuation, and formatting. Reject omitted
supported details, hallucinated details, summaries, over-specific rewrites, and empty content.
For a partial input, reject any target containing content beyond the observed semantic boundary.
Already clean passthrough input is valid when unchanged. Return strict JSON only:
{"pass":true,"reason":"brief reason"}."""

SEMANTIC_QC_USER = """Input transcript:
{input}

Refined target:
{refined}

Does the target exactly preserve the complete final intended meaning? Return JSON only."""


async def qc_one(item: dict, llm: AsyncLLMClient, sem: asyncio.Semaphore) -> tuple[dict, bool, str]:
    inp = item.get("input", "")
    refined = item.get("output", {}).get("refined_text", "")

    if not inp or not refined:
        return item, False, "input or refined_text is empty"

    # Large expansions are likely unsupported.
    if len(refined) > len(inp) * 1.8 and len(refined) - len(inp) > 30:
        return item, False, f"abnormal expansion: input={len(inp)} refined={len(refined)}"

    # Reject content-free fragments.
    if len(inp.strip()) <= 2:
        return item, False, f"input is too short to contain substantive content: {inp!r}"

    async with sem:
        try:
            user = SEMANTIC_QC_USER.format(input=inp, refined=refined)
            response = await llm.generate(
                system_prompt=SEMANTIC_QC_SYSTEM,
                user_prompt=user,
                temperature=0.1,
                max_tokens=LLM_MAX_TOKENS_QC,
            )
        except Exception as e:
            logger.warning("Semantic QC failed: %s", e)
            return item, False, "QC call failed"

    try:
        result = json.loads(response.strip())
        if isinstance(result, dict):
            passed = result.get("pass", True)
            reason = result.get("reason", "")
            return item, passed, reason
    except (json.JSONDecodeError, ValueError):
        pass

    # Accept simple boolean fallbacks from otherwise non-conforming responses.
    lowered = response.strip().lower()
    if lowered.startswith("true") or lowered.startswith('{"pass": true'):
        return item, True, response[:100]
    elif lowered.startswith("false") or lowered.startswith('{"pass": false'):
        return item, False, response[:100]

    logger.debug("QC unparseable verdict: %s", response[:100])
    return item, False, "unparseable QC response"


async def run_qc_scene(
    items: list[dict],
    target: int,
    concurrent: int,
    base_url: str | None = None,
) -> tuple[list[dict], list[dict], int]:
    """Check one scene until the requested number of records passes."""
    client_kwargs = {"max_concurrent": concurrent}
    if base_url:
        client_kwargs["base_url"] = base_url
    llm = AsyncLLMClient(**client_kwargs)
    sem = asyncio.Semaphore(concurrent)

    passed, rejected = [], []
    remaining = list(items)
    checked = 0
    pbar = tqdm(total=target, desc="QC", unit="item")

    while len(passed) < target and remaining:
        batch = remaining[:concurrent * 2]
        del remaining[:concurrent * 2]

        tasks = [qc_one(it, llm, sem) for it in batch]
        batch_results = await asyncio.gather(*tasks)
        checked += len(batch_results)

        for item, ok, reason in batch_results:
            if ok:
                passed.append(item)
                pbar.update(1)
                if len(passed) >= target:
                    break
            else:
                item["qc_reason"] = reason
                rejected.append(item)

    pbar.close()
    return passed, rejected, checked


def load_by_scene(processed_dir: Path) -> dict[str, list[dict]]:
    by_scene: dict[str, list[dict]] = {}
    for jsonl_file in sorted(processed_dir.glob("*.jsonl")):
        scene = jsonl_file.stem
        records = read_jsonl(jsonl_file)
        by_scene[scene] = records
        logger.info("Loaded %d records from %s", len(records), jsonl_file.name)
    return by_scene


def validate_distribution(stats: dict) -> list[str]:
    warnings = []
    total = stats["total"]
    if total == 0:
        return warnings
    for scene, target_ratio in SCENE_DISTRIBUTION.items():
        actual_count = stats["scene"].get(scene, 0)
        actual_ratio = actual_count / total
        deviation = abs(actual_ratio - target_ratio)
        if deviation > 0.05:
            warnings.append(
                f"{scene}: target={target_ratio:.2%}, "
                f"actual={actual_ratio:.2%} (dev={deviation:.2%})"
            )
    return warnings


def main():
    parser = argparse.ArgumentParser(description="Finalize, validate, and deduplicate pairs")
    parser.add_argument("--total", type=int, default=TOTAL_SAMPLES, help="target record count")
    parser.add_argument("--threshold", type=float, default=DEDUP_THRESHOLD)
    parser.add_argument("--skip-qc", action="store_true", help="skip semantic QC")
    parser.add_argument("--qc-concurrent", type=int, default=LLM_MAX_CONCURRENT,
                        help="QC concurrency (default: %d)" % LLM_MAX_CONCURRENT)
    parser.add_argument("--base-url", type=str, default=None, help="LLM service URL")
    parser.add_argument("--save-rejects", action="store_true",
                        help="write rejected records to data/final/rejects.jsonl")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    processed_dir = Path(PROCESSED_DIR)
    if not processed_dir.exists():
        logger.error("Processed dir not found: %s", processed_dir)
        sys.exit(1)

    logger.info("=== Loading scene data ===")
    by_scene = load_by_scene(processed_dir)
    if not by_scene:
        logger.error("No data found in %s", processed_dir)
        sys.exit(1)

    # Compute per-scene targets.
    available_scenes = list(by_scene.keys())
    targets: dict[str, int] = {}
    for scene in available_scenes:
        ratio = SCENE_DISTRIBUTION.get(scene, 0.05)
        targets[scene] = max(1, int(args.total * ratio))
        logger.info("  %s: available=%d, target=%d (%.0f%%)",
                    scene, len(by_scene[scene]), targets[scene], ratio * 100)

    # Run semantic QC until each scene reaches its target.
    all_rejects = []
    if args.skip_qc:
        logger.info("=== Skipping Semantic QC ===")
        passed_data = []
        for scene in available_scenes:
            items = by_scene[scene][:targets[scene]]
            passed_data.extend(items)
            logger.info("  %s: took %d/%d (no QC)", scene, len(items), targets[scene])
    else:
        passed_data = []
        for scene in available_scenes:
            target = targets[scene]
            items = by_scene.get(scene, [])
            logger.info("=== QC: %s (need %d, available %d) ===", scene, target, len(items))
            passed, rejected, checked = asyncio.run(
                run_qc_scene(items, target, args.qc_concurrent, args.base_url)
            )
            passed_data.extend(passed)
            all_rejects.extend(rejected)
            shortfall = max(0, target - len(passed))
            logger.info("  %s: passed=%d, rejected=%d, checked=%d, shortfall=%d",
                        scene, len(passed), len(rejected), checked, shortfall)

        if args.save_rejects and all_rejects:
            rejects_path = Path(FINAL_DIR) / "rejects.jsonl"
            rejects_path.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(all_rejects, rejects_path)
            logger.info("Rejects saved: %s (%d records)", rejects_path, len(all_rejects))

    # Global near-duplicate filtering.
    logger.info("=== Global dedup (threshold=%.2f) ===", args.threshold)
    before = len(passed_data)
    final_data = deduplicate(passed_data, text_key="input", threshold=args.threshold)
    logger.info("Dedup: %d -> %d (removed %d, %.1f%%)",
                before, len(final_data), before - len(final_data),
                (before - len(final_data)) / before * 100 if before else 0)

    # Distribution validation.
    logger.info("=== Distribution validation ===")
    stats = compute_distribution(final_data)
    warnings = validate_distribution(stats)
    for msg in warnings:
        logger.warning("Distribution drift: %s", msg)
    if not warnings:
        logger.info("Distribution check passed.")

    # Write the final training set.
    logger.info("=== Writing final dataset ===")
    final_dir = Path(FINAL_DIR)
    final_dir.mkdir(parents=True, exist_ok=True)

    train_data = []
    for item in final_data:
        train_data.append({
            "input": item["input"],
            "output": item["output"],
            "meta": item.get("meta", {}),
        })

    train_path = final_dir / "train.jsonl"
    write_jsonl(train_data, train_path)
    logger.info("Final: %s (%d records)", train_path, len(train_data))

    stats_output = {
        "total_samples": stats["total"],
        "target_samples": args.total,
        "coverage": stats["total"] / args.total if args.total else 0,
        "dedup_threshold": args.threshold,
        "distribution": stats,
        "warnings": warnings,
    }
    stats_path = final_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_output, f, ensure_ascii=False, indent=2)
    logger.info("Stats: %s", stats_path)
    logger.info("DONE: %d / %d (%.1f%%)", stats["total"], args.total,
                stats["total"] / args.total * 100 if args.total else 0)


if __name__ == "__main__":
    main()
