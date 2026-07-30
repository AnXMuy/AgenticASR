# -*- coding: utf-8 -*-
"""Optional real TTS -> ASR backend.

The main pipeline should stay lightweight: Step 3 calls HTTP services for heavy
MOSS-TTS and Qwen3-ASR models, while text_sim remains the default fallback.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTSASRConfig:
    tts_provider: str
    stt_provider: str
    device: str
    cache_dir: Path
    enable_cache: bool = True
    strength: str = "medium"
    tts_base_url: str = "http://127.0.0.1:8011"
    stt_base_url: str = "http://127.0.0.1:8012"
    timeout: float = 600.0


class TTSASRBackend:
    """Real speech backend interface for oral_text -> ASR text."""

    def __init__(self, config: TTSASRConfig):
        self.config = config
        if self.config.enable_cache:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_key(self, text: str) -> str:
        payload = "\n".join([
            self.config.tts_provider,
            self.config.stt_provider,
            self.config.device,
            self.config.strength,
            self.config.tts_base_url,
            self.config.stt_base_url,
            text,
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cached_audio_path(self, text: str) -> Path:
        return self.config.cache_dir / f"{self.cache_key(text)}.wav"

    def _cached_text_path(self, text: str) -> Path:
        return self.config.cache_dir / f"{self.cache_key(text)}.txt"

    async def transcribe(self, text: str, language: str | None = None) -> str:
        """Generate speech from text and transcribe it back to text."""
        if self.config.tts_provider != "moss_api":
            raise NotImplementedError(
                f"Unsupported TTS provider: {self.config.tts_provider}. "
                "Use ASR_TTS_PROVIDER=moss_api or --tts-provider moss_api."
            )
        if self.config.stt_provider != "qwen3_api":
            raise NotImplementedError(
                f"Unsupported STT provider: {self.config.stt_provider}. "
                "Use ASR_STT_PROVIDER=qwen3_api or --stt-provider qwen3_api."
            )

        text_cache = self._cached_text_path(text)
        if self.config.enable_cache and text_cache.exists():
            cached = text_cache.read_text(encoding="utf-8").strip()
            if cached:
                return cached

        audio_path = self._cached_audio_path(text)
        if not (self.config.enable_cache and audio_path.exists()):
            audio_path = await self._synthesize_to_file(text=text, output_path=audio_path, language=language)

        transcript = await self._transcribe_file(audio_path=audio_path, language=language)
        transcript = transcript.strip()
        if not transcript:
            raise RuntimeError("Qwen3-ASR returned an empty transcript.")
        if self.config.enable_cache:
            text_cache.write_text(transcript, encoding="utf-8")
        return transcript

    async def _synthesize_to_file(self, text: str, output_path: Path, language: str | None = None) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "text": text,
            "output_path": str(output_path),
            "strength": self.config.strength,
        }
        if language:
            payload["language"] = _normalize_language(language)
        url = self.config.tts_base_url.rstrip("/") + "/v1/tts"
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        path = Path(data.get("audio_path") or output_path)
        if not path.exists():
            raise RuntimeError(f"MOSS-TTS service did not create audio file: {path}")
        return path

    async def _transcribe_file(self, audio_path: Path, language: str | None = None) -> str:
        payload: dict[str, Any] = {"audio_path": str(audio_path)}
        if language:
            payload["language"] = _normalize_language(language)
        url = self.config.stt_base_url.rstrip("/") + "/v1/asr"
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        return str(data.get("text") or "")


def _normalize_language(language: str) -> str:
    lang = (language or "").strip().lower()
    if lang in {"zh", "cn", "chinese"}:
        return "Chinese"
    if lang in {"en", "eng", "english"}:
        return "English"
    return language
