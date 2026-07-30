# -*- coding: utf-8 -*-
"""Export final records to stratified LLaMAFactory ShareGPT datasets."""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.io_utils import read_jsonl

DEFAULT_VALIDATION_RATIO = 0.15

SYSTEM_PROMPT = (
    "你是 ASR 文本纠错助手。保留原意，最小修改：去口癖/重复，修错字，补必要标点，"
    "规范数字、日期、术语和代码符号，处理自我修正。不要总结、扩写或解释。"
    "重要易错实体在末尾追加 <KEY>[词1、词2]；没有则不加。"
)


def convert_one(item: dict) -> dict | None:
    """Convert one finalized record to ShareGPT format."""
    input_text = item.get("input", "")
    output = item.get("output", {})
    refined_text = output.get("refined_text", "")

    if not input_text or not refined_text:
        return None

    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": input_text},
            {"from": "gpt", "value": refined_text},
        ]
    }


def get_length_bucket(text: str) -> str:
    """Assign a length bucket from the refined target."""
    length = len(text)
    if length <= 20:
        return "short"
    elif length <= 50:
        return "medium"
    elif length <= 100:
        return "long"
    else:
        return "very_long"


def stratified_split(
    samples: list[dict],
    val_split: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Split by scene and target length while retaining group coverage."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in samples:
        scene = sample.get("_scene", "unknown")
        length_bucket = sample.get("_length_bucket", "medium")
        groups[(scene, length_bucket)].append(sample)

    train_samples = []
    val_samples = []

    for (scene, length_bucket), group_samples in groups.items():
        rng.shuffle(group_samples)
        split_n = int(len(group_samples) * val_split)

        # Put at least one sufficiently represented group item in validation.
        if len(group_samples) >= 5 and split_n == 0:
            split_n = 1

        group_val = group_samples[:split_n]
        group_train = group_samples[split_n:]

        val_samples.extend(group_val)
        train_samples.extend(group_train)

    # Shuffle after group-wise splitting.
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    return train_samples, val_samples


def main():
    parser = argparse.ArgumentParser(description="Export LLaMAFactory ShareGPT datasets")
    parser.add_argument("--inputs", type=str, nargs="+",
                        default=["data/final/train.jsonl"],
                        help="one or more finalized JSONL files")
    parser.add_argument("--train-output", type=str, default="data/final/train_sft.json")
    parser.add_argument("--val-output", type=str, default="data/final/val_sft.json")
    parser.add_argument("--val-split", type=float, default=DEFAULT_VALIDATION_RATIO,
                        help=f"validation ratio (default: {DEFAULT_VALIDATION_RATIO})")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no-system", action="store_true",
                        help="omit the system prompt")
    parser.add_argument("--no-stratified", action="store_true",
                        help="use a simple random split")
    args = parser.parse_args()

    # Load all requested datasets.
    all_data = []
    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"[WARN] Input file not found, skipping: {input_path}")
            continue

        data = read_jsonl(input_path)
        print(f"Loaded {len(data)} samples from {input_path}")
        all_data.extend(data)

    if not all_data:
        print("[ERROR] No data loaded from any input file")
        sys.exit(1)

    print(f"Total samples: {len(all_data)}")

    # Convert records.
    samples = []
    skipped = 0
    for item in all_data:
        conv = convert_one(item)
        if conv is None:
            skipped += 1
            continue
        if args.no_system:
            conv["conversations"] = conv["conversations"][1:]  # remove system
        # Temporary metadata supports stratification.
        meta = item.get("meta", {})
        conv["_scene"] = meta.get("scene", "unknown")
        conv["_length_bucket"] = get_length_bucket(item.get("output", {}).get("refined_text", ""))

        samples.append(conv)

    print(f"Converted: {len(samples)}, skipped: {skipped}")

    rng = random.Random(args.seed)

    if args.no_stratified:
        # Simple random split.
        rng.shuffle(samples)
        split_n = int(len(samples) * args.val_split)
        val_samples = samples[:split_n]
        train_samples = samples[split_n:]
    else:
        # Scene-and-length stratified split.
        train_samples, val_samples = stratified_split(samples, args.val_split, rng)

    # Remove temporary metadata before export.
    for sample in train_samples:
        sample.pop("_scene", None)
        sample.pop("_length_bucket", None)
    for sample in val_samples:
        sample.pop("_scene", None)
        sample.pop("_length_bucket", None)

    # Print summary statistics.
    print("\n=== Dataset Statistics ===")
    print(f"Train: {len(train_samples)} samples")
    print(f"Val:   {len(val_samples)} samples")
    print(f"Split: {args.val_split:.0%} validation")

    # Write datasets.
    for path, subset in [
        (Path(args.train_output), train_samples),
        (Path(args.val_output), val_samples),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"  {path}: {len(subset)} samples")

    print(f"\nDone. train={len(train_samples)}, val={len(val_samples)}")


if __name__ == "__main__":
    main()
