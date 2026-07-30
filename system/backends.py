"""Audio, VAD, and sherpa-onnx backends for the terminal demo."""

from __future__ import annotations

import glob
import os
import queue
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
VAD_WINDOW = 512

_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_PUNCT = "\\u3000-\\u303f\\uff00-\\uffef"
_ASCII_PUNCT = re.escape(",.!?;:%)]}")


def normalize_cjk(text: str) -> str:
    """Remove ASR-inserted spaces around CJK characters and punctuation."""

    text = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", text)
    text = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK_PUNCT}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_PUNCT}])\s+(?=[{_CJK}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_PUNCT}])\s+(?=[{_CJK_PUNCT}])", "", text)
    return re.sub(rf"\s+(?=[{_ASCII_PUNCT}])", "", text)


class EnergyVad:
    """Dependency-free energy VAD intended for diagnostics."""

    def __init__(
        self,
        threshold: float = 0.02,
        min_silence: float = 0.5,
        min_speech: float = 0.2,
    ) -> None:
        self.threshold = threshold
        self.min_silence = min_silence
        self.min_speech = min_speech
        self.in_speech = False
        self.speech_run = 0.0
        self.silence_run = 0.0
        self._pending = 0

    def accept_waveform(self, window: np.ndarray) -> None:
        duration = len(window) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(np.asarray(window, dtype=np.float32) ** 2)) + 1e-9)
        if rms > self.threshold:
            self.speech_run += duration
            self.silence_run = 0.0
            if not self.in_speech and self.speech_run >= self.min_speech:
                self.in_speech = True
        else:
            self.silence_run += duration
            self.speech_run = 0.0
            if self.in_speech and self.silence_run >= self.min_silence:
                self.in_speech = False
                self._pending += 1

    def is_speech_detected(self) -> bool:
        return self.in_speech

    def empty(self) -> bool:
        return self._pending == 0

    def pop(self) -> None:
        self._pending = max(0, self._pending - 1)


class FireRedVad:
    """Adapter for the optional FireRedVAD streaming backend."""

    def __init__(
        self,
        model_dir: str,
        speech_threshold: float = 0.5,
        min_silence: float = 0.7,
        min_speech: float = 0.2,
        chunk_seconds: float = 0.3,
    ) -> None:
        from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig

        frames_per_second = 100
        config = FireRedStreamVadConfig(
            speech_threshold=speech_threshold,
            min_speech_frame=max(1, round(min_speech * frames_per_second)),
            min_silence_frame=max(1, round(min_silence * frames_per_second)),
        )
        self.vad = FireRedStreamVad.from_pretrained(model_dir, config)
        self.vad.reset()
        self.chunk_size = int(chunk_seconds * SAMPLE_RATE)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.in_speech = False
        self._pending = 0

    def accept_waveform(self, window: np.ndarray) -> None:
        self.buffer = np.concatenate([self.buffer, np.asarray(window, dtype=np.float32)])
        while len(self.buffer) >= self.chunk_size:
            waveform = (np.clip(self.buffer[: self.chunk_size], -1.0, 1.0) * 32767).astype(
                np.int16
            )
            for result in self.vad.detect_chunk(waveform):
                if result.is_speech_start:
                    self.in_speech = True
                if result.is_speech_end:
                    self.in_speech = False
                    self._pending += 1
            self.buffer = self.buffer[self.chunk_size :]

    def is_speech_detected(self) -> bool:
        return self.in_speech

    def empty(self) -> bool:
        return self._pending == 0

    def pop(self) -> None:
        self._pending = max(0, self._pending - 1)


def find_asr_files(asr_dir: str) -> tuple[str, dict[str, str]]:
    """Discover a sherpa-onnx transducer or Wenet CTC checkpoint."""

    models = sorted(glob.glob(os.path.join(asr_dir, "*.onnx")))
    models = [path for path in models if "vad" not in Path(path).name.lower()]
    if not models:
        raise FileNotFoundError(f"no ONNX ASR model found in {asr_dir}")
    tokens = os.path.join(asr_dir, "tokens.txt")
    if not os.path.isfile(tokens):
        raise FileNotFoundError(f"missing token table: {tokens}")

    def pick(fragment: str) -> str | None:
        candidates = [path for path in models if fragment in Path(path).name.lower()]
        unquantized = [path for path in candidates if "int8" not in Path(path).name.lower()]
        return (unquantized or candidates or [None])[0]

    encoder, decoder, joiner = pick("encoder"), pick("decoder"), pick("joiner")
    if encoder and decoder and joiner:
        return "transducer", {
            "tokens": tokens,
            "encoder": encoder,
            "decoder": decoder,
            "joiner": joiner,
        }
    model = pick("model") or pick("ctc") or encoder
    if model is None:
        raise FileNotFoundError(f"could not identify an ASR checkpoint in {asr_dir}")
    return "wenet-ctc", {"tokens": tokens, "model": model}


def build_recognizer(
    asr_dir: str,
    asr_type: str,
    provider: str,
    model_type: str = "",
):
    """Create a sherpa-onnx online recognizer."""

    try:
        import sherpa_onnx
    except ImportError as error:
        raise RuntimeError("streaming ASR requires `pip install sherpa-onnx`") from error

    detected_type, files = find_asr_files(asr_dir)
    kind = detected_type if asr_type == "auto" else asr_type
    common = {
        "num_threads": 2,
        "provider": provider,
        "decoding_method": "greedy_search",
        "enable_endpoint_detection": False,
    }
    if kind == "transducer":
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            **files, model_type=model_type, **common
        )
    if kind == "wenet-ctc":
        return sherpa_onnx.OnlineRecognizer.from_wenet_ctc(
            tokens=files["tokens"],
            model=files["model"],
            chunk_size=16,
            num_left_chunks=4,
            **common,
        )
    raise ValueError(f"unsupported ASR type: {kind}")


def build_vad(
    kind: str,
    vad_model: str,
    firered_dir: str,
    threshold: float,
    min_silence: float,
    min_speech: float,
    energy_threshold: float,
    provider: str,
):
    """Create the selected voice activity detector."""

    if kind == "energy":
        return EnergyVad(energy_threshold, min_silence, min_speech)
    if kind == "firered":
        return FireRedVad(firered_dir, threshold, min_silence, min_speech)

    try:
        import sherpa_onnx
    except ImportError as error:
        raise RuntimeError("Silero VAD requires `pip install sherpa-onnx`") from error
    if not os.path.isfile(vad_model):
        raise FileNotFoundError(f"VAD model not found: {vad_model}")
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = vad_model
    config.silero_vad.threshold = threshold
    config.silero_vad.min_silence_duration = min_silence
    config.silero_vad.min_speech_duration = min_speech
    config.silero_vad.window_size = VAD_WINDOW
    config.sample_rate = SAMPLE_RATE
    config.provider = provider
    return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)


def iter_wav(path: str) -> Iterator[np.ndarray]:
    """Yield mono 16 kHz windows from an audio file."""

    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError("audio-file input requires `pip install soundfile`") from error
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        output_size = round(len(mono) * SAMPLE_RATE / sample_rate)
        mono = np.interp(
            np.linspace(0, 1, output_size), np.linspace(0, 1, len(mono)), mono
        ).astype("float32")
    for offset in range(0, len(mono), VAD_WINDOW):
        window = mono[offset : offset + VAD_WINDOW]
        if len(window) < VAD_WINDOW:
            window = np.pad(window, (0, VAD_WINDOW - len(window)))
        yield window
    for _ in range(round(SAMPLE_RATE / VAD_WINDOW)):
        yield np.zeros(VAD_WINDOW, dtype="float32")


def iter_microphone(device_index: int | None) -> Iterator[np.ndarray]:
    """Yield windows from a microphone until interrupted."""

    try:
        import sounddevice as sd
    except ImportError as error:
        raise RuntimeError("microphone input requires `pip install sounddevice`") from error
    windows: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, frames, timing, status) -> None:  # noqa: ANN001, ARG001
        if status:
            print(status, file=sys.stderr)
        windows.put(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=VAD_WINDOW,
        device=device_index,
        callback=callback,
    ):
        while True:
            yield windows.get()
