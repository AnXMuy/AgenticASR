"""Terminal entry point for streaming AgenticASR inference."""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .backends import (
    SAMPLE_RATE,
    VAD_WINDOW,
    build_recognizer,
    build_vad,
    iter_microphone,
    iter_wav,
    normalize_cjk,
)
from .chunking import ChunkManager
from .refiner import IdentityRefiner, MLXRefiner, StreamingRefinementSession


def run_stream(
    recognizer,
    vad,
    windows: Iterable[np.ndarray],
    session: StreamingRefinementSession,
    *,
    max_chunk_chars: int = 80,
    tail_pad_seconds: float = 1.0,
    preroll_seconds: float = 0.7,
    normalize: bool = True,
) -> None:
    """Decode an audio stream and print replaceable refined transcripts."""

    chunk_manager = ChunkManager(max_chars=max_chunk_chars)
    stream = None
    active = False
    preroll: list[np.ndarray] = []
    preroll_windows = max(1, round(preroll_seconds * SAMPLE_RATE / VAD_WINDOW))

    def format_text(text: str) -> str:
        return normalize_cjk(text) if normalize else text

    def emit(hypothesis: str, *, vad_boundary: bool) -> None:
        for chunk in chunk_manager.update(
            format_text(hypothesis), vad_boundary=vad_boundary
        ):
            update = session.add(chunk)
            print(
                f"[refined window={len(update.raw_chunks)} "
                f"latency={update.latency_ms:.0f}ms] {update.transcript}"
            )

    def finalize() -> None:
        nonlocal active, stream
        stream.accept_waveform(
            SAMPLE_RATE, np.zeros(round(tail_pad_seconds * SAMPLE_RATE), dtype="float32")
        )
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        emit(recognizer.get_result(stream), vad_boundary=True)
        active = False
        stream = None

    try:
        for window in windows:
            vad.accept_waveform(window)
            speech = vad.is_speech_detected()
            if speech and not active:
                active = True
                stream = recognizer.create_stream()
                for buffered in preroll:
                    stream.accept_waveform(SAMPLE_RATE, buffered)
            if active:
                stream.accept_waveform(SAMPLE_RATE, window)
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                partial = format_text(recognizer.get_result(stream))
                if partial:
                    print(f"\r[partial] {partial}", end="", flush=True)
                    emit(partial, vad_boundary=False)
            if active and not speech:
                print()
                finalize()
            preroll.append(window)
            if len(preroll) > preroll_windows:
                preroll.pop(0)
            while not vad.empty():
                vad.pop()
    finally:
        if active and stream is not None:
            print()
            finalize()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_provider = "coreml" if sys.platform == "darwin" and platform.machine() == "arm64" else "cpu"
    parser = argparse.ArgumentParser(
        description="Streaming AgenticASR: online ASR, VAD chunking, and windowed refinement"
    )
    parser.add_argument("--asr-dir", type=Path, default=Path("models/asr"))
    parser.add_argument(
        "--asr-type", choices=("auto", "transducer", "wenet-ctc"), default="auto"
    )
    parser.add_argument("--model-type", default="")
    parser.add_argument("--provider", default=default_provider)
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--vad", choices=("silero", "energy", "firered"), default="silero")
    parser.add_argument("--vad-model", type=Path, default=Path("models/silero_vad.onnx"))
    parser.add_argument("--firered-dir", type=Path, default=Path("models/firered_vad"))
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-silence", type=float, default=0.7)
    parser.add_argument("--vad-min-speech", type=float, default=0.25)
    parser.add_argument("--energy-threshold", type=float, default=0.02)
    parser.add_argument("--tail-pad", type=float, default=1.0)
    parser.add_argument("--preroll", type=float, default=0.7)
    parser.add_argument("--max-chunk-chars", type=int, default=80)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--refiner", type=Path, help="MLX Refiner checkpoint directory")
    parser.add_argument(
        "--identity-refiner",
        action="store_true",
        help="diagnostic mode: bypass Refiner and print ASR/chunking output",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-cjk-normalize", action="store_true")
    args = parser.parse_args(argv)
    if args.refiner and args.identity_refiner:
        parser.error("--refiner and --identity-refiner are mutually exclusive")
    if not args.list_devices and not args.refiner and not args.identity_refiner:
        parser.error(
            "--refiner is required for the AgenticASR app; "
            "use --identity-refiner only for ASR/chunking diagnostics"
        )
    if args.max_chunk_chars < 1:
        parser.error("--max-chunk-chars must be at least 1")
    if args.window_size < 1:
        parser.error("--window-size must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_devices:
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError("device listing requires `pip install sounddevice`") from error
        print(sd.query_devices())
        return 0

    refiner = (
        IdentityRefiner()
        if args.identity_refiner
        else MLXRefiner(str(args.refiner), max_tokens=args.max_new_tokens)
    )
    session = StreamingRefinementSession(refiner, window_size=args.window_size)
    recognizer = build_recognizer(
        str(args.asr_dir), args.asr_type, args.provider, args.model_type
    )
    vad = build_vad(
        args.vad,
        str(args.vad_model),
        str(args.firered_dir),
        args.vad_threshold,
        args.vad_min_silence,
        args.vad_min_speech,
        args.energy_threshold,
        args.provider,
    )
    windows = iter_wav(str(args.wav)) if args.wav else iter_microphone(args.device_index)
    run_stream(
        recognizer,
        vad,
        windows,
        session,
        max_chunk_chars=args.max_chunk_chars,
        tail_pad_seconds=args.tail_pad,
        preroll_seconds=args.preroll,
        normalize=not args.no_cjk_normalize,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
