# -*- coding: utf-8 -*-
"""Step 4: materialize the common training-record schema."""
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import KEYWORD_MODE, PROCESSED_DIR, RAW_DIR, TOTAL_SAMPLES
from src.utils.io_utils import read_jsonl, setup_logging, write_jsonl
from src.utils.keyword_utils import extract_keywords, normalize_review_targets


def _min_length(scene: str) -> int:
    if scene.startswith("english_"):
        return 6
    if scene in ("vibe_coding", "academic", "dictation_memo", "voice_search", "passthrough"):
        return 8
    return 10



def _lang_for(scene: str, item: dict) -> str:
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    return item.get("language") or meta.get("language") or ("en" if scene.startswith("english_") else "zh")


def _assemble_from_asr(scene: str, raw_data: list[dict]) -> tuple[list[dict], dict[str, int]]:
    assembled = []
    stats = {"skipped_empty": 0, "skipped_short": 0}
    min_len = _min_length(scene)

    for item in raw_data:
        input_text = item.get("input", "")
        oral_text = item.get("oral_text", "")
        output = item.get("output") if isinstance(item.get("output"), dict) else {}
        clean_text = output.get("refined_text") or item.get("clean_text", "")
        keywords = normalize_review_targets(
            extract_keywords(item),
            input_text=oral_text or input_text,
            refined_text=clean_text,
        )

        if not input_text or not clean_text:
            stats["skipped_empty"] += 1
            continue
        if len(input_text.strip()) < 2 or len(clean_text.strip()) < min_len:
            stats["skipped_short"] += 1
            continue

        meta = dict(item.get("meta", {})) if isinstance(item.get("meta"), dict) else {}
        meta.setdefault("scene", scene)
        meta.setdefault("language", _lang_for(scene, item))
        meta.setdefault("speech_phenomena", [])
        meta.setdefault("itn_types", [])
        meta.setdefault("is_fragment", False)
        meta.setdefault("keyword_mode", KEYWORD_MODE)

        record = {
            "input": input_text,
            "oral_text": oral_text or input_text,
            "output": {
                "refined_text": clean_text,
                "keyword_list": keywords,
            },
            "meta": meta,
        }
        if "id" in item:
            record["id"] = item["id"]
        assembled.append(record)

    return assembled, stats


def _assemble_from_clean(scene: str, raw_data: list[dict]) -> tuple[list[dict], dict[str, int]]:
    assembled = []
    stats = {"skipped_empty": 0, "skipped_short": 0}
    min_len = _min_length(scene)

    for item in raw_data:
        oral_text = item.get("oral_text", "")
        clean_text = item.get("clean_text", "")
        keywords = normalize_review_targets(
            extract_keywords(item),
            input_text=oral_text,
            refined_text=clean_text,
        )
        speech_phenomena = item.get("speech_phenomena", [])
        itn_types = item.get("itn_types", [])

        if not oral_text or not clean_text:
            stats["skipped_empty"] += 1
            continue
        if len(oral_text.strip()) < 2 or len(clean_text.strip()) < min_len:
            stats["skipped_short"] += 1
            continue

        lang = _lang_for(scene, item)
        meta = dict(item.get("meta", {})) if isinstance(item.get("meta"), dict) else {}
        meta.update({
            "scene": scene,
            "language": lang,
            "speech_phenomena": speech_phenomena,
            "itn_types": itn_types,
            "is_fragment": bool(meta.get("is_fragment", False)),
            "asr_mode": meta.get("asr_mode", "clean_fallback"),
            "keyword_mode": meta.get("keyword_mode", KEYWORD_MODE),
        })

        record = {
            "input": oral_text,
            "oral_text": oral_text,
            "output": {
                "refined_text": clean_text,
                "keyword_list": keywords,
            },
            "meta": meta,
        }
        if "id" in item:
            record["id"] = item["id"]
        assembled.append(record)

    return assembled, stats


def _choose_source(scene_dir: Path, requested: str) -> tuple[Path | None, str]:
    asr_path = scene_dir / "asr_sim.jsonl"
    clean_path = scene_dir / "clean.jsonl"

    if requested == "asr":
        return (asr_path if asr_path.exists() else None), "asr"
    if requested == "clean":
        return (clean_path if clean_path.exists() else None), "clean"
    if asr_path.exists():
        return asr_path, "asr"
    if clean_path.exists():
        return clean_path, "clean"
    return None, requested


def assemble_scene(scene: str, input_source: str = "auto") -> list[dict]:
    """Load one scene, apply basic checks, and create training records."""
    logger = logging.getLogger(__name__)
    scene_dir = Path(RAW_DIR) / scene
    source_path, actual_source = _choose_source(scene_dir, input_source)

    if source_path is None:
        logger.error("No input data found for scene '%s' (requested=%s)", scene, input_source)
        return []

    logger.info("Scene '%s': assembling from %s", scene, source_path.name)
    raw_data = read_jsonl(source_path)
    if actual_source == "asr":
        assembled, stats = _assemble_from_asr(scene, raw_data)
    else:
        assembled, stats = _assemble_from_clean(scene, raw_data)

    logger.info(
        "Scene '%s': assembled %d from %s, skipped_empty=%d, skipped_short=%d",
        scene, len(assembled), source_path.name, stats["skipped_empty"], stats["skipped_short"],
    )
    return assembled


def main():
    parser = argparse.ArgumentParser(description="Materialize common training records")
    parser.add_argument("--scene", type=str, default=None, help="one internal scene key")
    parser.add_argument("--total", type=int, default=TOTAL_SAMPLES, help="compatibility option")
    parser.add_argument("--input-source", type=str, default="auto", choices=["auto", "asr", "clean"],
                        help="source: auto prefers asr_sim.jsonl, then clean.jsonl")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    output_dir = Path(PROCESSED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_assembled = 0

    if args.scene:
        logger.info("--- Assembling: %s ---", args.scene)
        records = assemble_scene(args.scene, input_source=args.input_source)
        if records:
            out_path = output_dir / f"{args.scene}.jsonl"
            write_jsonl(records, out_path)
            total_assembled += len(records)
            logger.info("Saved %d records to %s", len(records), out_path)
    else:
        raw_dir = Path(RAW_DIR)
        if not raw_dir.exists():
            logger.error("Raw dir not found: %s", raw_dir)
            return
        scenes = sorted([
            d.name for d in raw_dir.iterdir()
            if d.is_dir() and ((d / "asr_sim.jsonl").exists() or (d / "clean.jsonl").exists())
        ])
        for scene in scenes:
            logger.info("--- Assembling: %s ---", scene)
            records = assemble_scene(scene, input_source=args.input_source)
            if records:
                out_path = output_dir / f"{scene}.jsonl"
                write_jsonl(records, out_path)
                total_assembled += len(records)
                logger.info("  Saved %d records to %s", len(records), out_path)

    logger.info("All done. Total assembled: %d", total_assembled)


if __name__ == "__main__":
    main()
