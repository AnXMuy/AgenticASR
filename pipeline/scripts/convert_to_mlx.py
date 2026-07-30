#!/usr/bin/env python3
"""Convert a Hugging Face checkpoint to MLX, with optional quantization."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

SUPPORTED_QUANTS = ("q2_8", "q3_0", "q3_5", "q4_0", "q4_1", "q5_0", "q5_1", "q6_0", "q8_0", "bf16")
QUANT_ARGS = {
    "q2_8": ["-q", "--q-bits", "2", "--q-group-size", "8"],
    "q3_0": ["-q", "--q-bits", "3", "--q-group-size", "64"],
    "q3_5": ["-q", "--q-bits", "3", "--q-group-size", "32"],
    "q4_0": ["-q", "--q-bits", "4", "--q-group-size", "64"],
    "q4_1": ["-q", "--q-bits", "4", "--q-group-size", "32"],
    "q5_0": ["-q", "--q-bits", "5", "--q-group-size", "64"],
    "q5_1": ["-q", "--q-bits", "5", "--q-group-size", "32"],
    "q6_0": ["-q", "--q-bits", "6", "--q-group-size", "64"],
    "q8_0": ["-q", "--q-bits", "8", "--q-group-size", "64"],
    "bf16": [],
}


def check_environment() -> None:
    """Require MLX on macOS before conversion."""

    if platform.system() != "Darwin":
        raise RuntimeError(f"MLX requires macOS; current platform is {platform.system()}")
    try:
        import mlx.core  # noqa: F401
    except ImportError as error:
        raise RuntimeError("install MLX with `pip install mlx mlx-lm`") from error


def convert(input_path: Path, output_path: Path, quantize: str | None = None) -> None:
    """Invoke the current mlx-lm conversion command."""

    command = [
        sys.executable,
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        str(input_path.resolve()),
        "--mlx-path",
        str(output_path.resolve()),
    ]
    if quantize and quantize != "bf16":
        command.extend(QUANT_ARGS[quantize])
    print("Command:", " ".join(command))
    subprocess.run(command, check=True)

    files = list(output_path.rglob("*")) if output_path.is_dir() else [output_path]
    files = [path for path in files if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"Converted {len(files)} files ({total_bytes / 1024**3:.2f} GiB) to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Hugging Face model to MLX")
    parser.add_argument("-i", "--input", type=Path, help="Hugging Face checkpoint directory")
    parser.add_argument("-o", "--output", type=Path, help="MLX output directory")
    parser.add_argument("-q", "--quantize", choices=SUPPORTED_QUANTS)
    parser.add_argument("--list-quants", action="store_true")
    parser.add_argument("--skip-check", action="store_true", help="skip the platform check")
    args = parser.parse_args()
    if args.list_quants:
        return args
    if args.input is None or args.output is None:
        parser.error("--input and --output are required")
    if not args.input.exists():
        parser.error(f"input path does not exist: {args.input}")
    return args


def main() -> int:
    args = parse_args()
    if args.list_quants:
        print("\n".join(SUPPORTED_QUANTS))
        return 0
    if not args.skip_check:
        check_environment()
    convert(args.input, args.output, args.quantize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
