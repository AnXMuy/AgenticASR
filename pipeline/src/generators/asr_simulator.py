# -*- coding: utf-8 -*-
"""Build aligned ASR-style input and Clean-target pairs for Step 3."""
import json
import logging
import random
import re
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import (
    ASR_CACHE_DIR,
    ASR_DEVICE,
    ASR_ENABLE_CACHE,
    ASR_FALLBACK_TEXT_SIM,
    ASR_HTTP_TIMEOUT,
    ASR_MODE,
    ASR_SIMULATION_RATIO,
    ASR_STRENGTH,
    ASR_STT_BASE_URL,
    ASR_STT_PROVIDER,
    ASR_TTS_BASE_URL,
    ASR_TTS_PROVIDER,
    KEYWORD_MODE,
    RAW_DIR,
)
from src.services.llm_client import AsyncLLMClient, LLMClient
from src.utils.io_utils import Checkpoint, append_jsonl, read_jsonl
from src.utils.keyword_utils import extract_keywords, normalize_review_targets

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Punctuation removed by the diagnostic rule-based simulator.
_REMOVABLE_PUNCT = set(
    "\u201c\u201d\u2018\u2019\u300c\u300d\u300e\u300f"
    "\uff08\uff09\u3010\u3011\u300a\u300b\u3008\u3009"
    "\u2014\u2026\uff1a\uff1b\u3001\u00b7\u2015()[]{}\"'"
)

_ALLOWED_STRENGTHS = {"light", "medium", "heavy"}
_ALLOWED_MODES = {"llm_sim", "text_sim", "tts_asr"}

ASR_SIMULATION_SYSTEM = """You create aligned partial training pairs for an ASR transcript Refiner.
Given an Oral transcript and its Clean target:
1. Choose a natural truncation boundary before the utterance ends.
2. Truncate both texts at the same semantic boundary. The target must not contain content that
   has not yet appeared in the truncated Oral text.
3. Corrupt only the truncated Oral text with plausible ASR omissions, substitutions, recognition
   errors, and irregular punctuation. Preserve enough evidence for its aligned Clean target.
Return strict JSON only: {"input": "...", "target": "..."}. Do not add explanations."""

ASR_SIMULATION_USER = """Oral transcript:
{oral}

Clean target:
{clean}

Return one aligned truncated-and-corrupted pair."""


def _remove_punct(text: str, strength: str = "medium") -> str:
    """Remove selected punctuation for diagnostic ASR degradation."""
    if strength == "light":
        keep = {"\uff0c", "\u3002", "\uff01", "\uff1f", ",", ".", "!", "?", "\uff1a", ":"}
    elif strength == "heavy":
        keep = {",", "."}
    else:
        keep = {"\uff0c", "\u3002", "\uff01", "\uff1f", ",", ".", "!", "?"}

    chars = []
    for c in text:
        if c in keep:
            chars.append(c)
        elif c in _REMOVABLE_PUNCT:
            chars.append(" ")
        else:
            chars.append(c)
    return "".join(chars)


def simulate_asr(text: str, strength: str = "medium") -> str:
    """Apply lightweight rule-based ASR degradation for diagnostics."""
    if strength not in _ALLOWED_STRENGTHS:
        strength = "medium"
    text = _remove_punct(text, strength=strength)
    text = re.sub(r"\s+", " ", text).strip()
    return text



class ASRSimulator:
    """Convert Clean records to aligned Refiner training records."""

    def __init__(
        self,
        async_client=None,
        mode: str | None = None,
        strength: str | None = None,
        tts_provider: str | None = None,
        stt_provider: str | None = None,
        tts_base_url: str | None = None,
        stt_base_url: str | None = None,
        device: str | None = None,
        enable_cache: bool | None = None,
        fallback_text_sim: bool | None = None,
        simulation_ratio: float = ASR_SIMULATION_RATIO,
        rng: random.Random | None = None,
    ):
        self.raw_dir = Path(RAW_DIR)
        self.mode = mode or ASR_MODE
        self.strength = strength or ASR_STRENGTH
        self.tts_provider = tts_provider or ASR_TTS_PROVIDER
        self.stt_provider = stt_provider or ASR_STT_PROVIDER
        self.tts_base_url = tts_base_url or ASR_TTS_BASE_URL
        self.stt_base_url = stt_base_url or ASR_STT_BASE_URL
        self.device = device or ASR_DEVICE
        self.enable_cache = ASR_ENABLE_CACHE if enable_cache is None else enable_cache
        self.fallback_text_sim = ASR_FALLBACK_TEXT_SIM if fallback_text_sim is None else fallback_text_sim
        self.simulation_ratio = simulation_ratio
        self.rng = rng or random.Random()
        self.llm = async_client
        self._backend = None
        if self.mode not in _ALLOWED_MODES:
            raise ValueError(f"Unsupported ASR mode: {self.mode}. Allowed: {sorted(_ALLOWED_MODES)}")
        if self.strength not in _ALLOWED_STRENGTHS:
            raise ValueError(f"Unsupported ASR strength: {self.strength}. Allowed: {sorted(_ALLOWED_STRENGTHS)}")
        if not 0 <= self.simulation_ratio <= 1:
            raise ValueError("simulation_ratio must be between 0 and 1")

    def _get_lang(self, scene: str) -> str:
        return "en" if scene.startswith("english_") else "zh"

    def _get_backend(self):
        if self._backend is None:
            from src.speech.tts_asr_backend import TTSASRBackend, TTSASRConfig

            self._backend = TTSASRBackend(TTSASRConfig(
                tts_provider=self.tts_provider,
                stt_provider=self.stt_provider,
                device=self.device,
                cache_dir=Path(ASR_CACHE_DIR),
                enable_cache=self.enable_cache,
                strength=self.strength,
                tts_base_url=self.tts_base_url,
                stt_base_url=self.stt_base_url,
                timeout=ASR_HTTP_TIMEOUT,
            ))
        return self._backend

    async def _simulate_with_llm(self, oral_text: str, clean_text: str) -> tuple[str, str]:
        if self.llm is None:
            self.llm = AsyncLLMClient()
        response = await self.llm.generate(
            system_prompt=ASR_SIMULATION_SYSTEM,
            user_prompt=ASR_SIMULATION_USER.format(oral=oral_text, clean=clean_text),
            temperature=0.7,
            max_tokens=1024,
        )
        try:
            parsed = LLMClient._parse_json(response)
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError("ASR simulator returned invalid JSON") from error
        if not isinstance(parsed, dict):
            raise ValueError("ASR simulator response must be a JSON object")
        input_text = parsed.get("input")
        target_text = parsed.get("target")
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("ASR simulator response is missing `input`")
        if not isinstance(target_text, str) or not target_text.strip():
            raise ValueError("ASR simulator response is missing `target`")
        return input_text.strip(), target_text.strip()

    async def _generate_pair(
        self,
        oral_text: str,
        clean_text: str,
        *,
        language: str | None = None,
        passthrough: bool = False,
    ) -> tuple[str, str, bool, bool]:
        if passthrough:
            return clean_text, clean_text, False, False
        selected = self.rng.random() < self.simulation_ratio
        if self.mode == "llm_sim":
            if selected:
                input_text, target_text = await self._simulate_with_llm(oral_text, clean_text)
                return input_text, target_text, True, False
            return oral_text, clean_text, False, False
        if self.mode == "text_sim":
            return simulate_asr(oral_text, strength=self.strength), clean_text, False, False
        try:
            input_text = await self._get_backend().transcribe(oral_text, language=language)
            return input_text, clean_text, False, False
        except Exception as e:
            if not self.fallback_text_sim:
                raise
            logger.warning("tts_asr failed; fallback to text_sim: %s", e)
            return simulate_asr(oral_text, strength=self.strength), clean_text, False, True

    async def simulate_one(self, scene: str, item: dict) -> dict:
        """Generate one ASR-style input and aligned target."""
        oral_text = item.get("oral_text", "")
        clean_text = item.get("clean_text") or item.get("output", {}).get("refined_text", "")
        lang = item.get("language") or self._get_lang(scene)

        if not oral_text or not clean_text:
            return {}

        speech_phenomena = item.get("speech_phenomena") or item.get("meta", {}).get("speech_phenomena", [])
        passthrough = "clean" in speech_phenomena
        asr_output, aligned_target, is_fragment, used_fallback = await self._generate_pair(
            oral_text,
            clean_text,
            language=lang,
            passthrough=passthrough,
        )
        keywords = normalize_review_targets(
            extract_keywords(item),
            input_text=oral_text,
            refined_text=aligned_target,
        )
        itn_types = item.get("itn_types") or item.get("meta", {}).get("itn_types", [])

        meta = dict(item.get("meta", {})) if isinstance(item.get("meta"), dict) else {}
        meta.update({
            "scene": scene,
            "language": lang,
            "speech_phenomena": speech_phenomena,
            "itn_types": itn_types,
            "is_fragment": is_fragment,
            "asr_mode": self.mode,
            "asr_strength": self.strength,
            "keyword_mode": KEYWORD_MODE,
            "tts_provider": self.tts_provider if self.mode == "tts_asr" else None,
            "stt_provider": self.stt_provider if self.mode == "tts_asr" else None,
            "asr_fallback_text_sim": used_fallback,
            "passthrough": passthrough,
        })

        result = {
            "input": asr_output,
            "oral_text": oral_text,
            "clean_text": aligned_target,
            "output": {
                "refined_text": aligned_target,
                "keyword_list": keywords,
            },
            "meta": meta,
        }
        if "id" in item:
            result["id"] = item["id"]
        return result

    async def simulate_scene(self, scene: str, resume: bool = True, input_source: str = "clean") -> int:
        """Generate ASR-style inputs for one scene with checkpointing."""
        scene_dir = self.raw_dir / scene
        source_name = "clean.jsonl" if input_source == "clean" else "oral.jsonl"
        source_path = scene_dir / source_name
        asr_path = scene_dir / "asr_sim.jsonl"
        ckpt_path = scene_dir / "asr_sim.ckpt.json"

        if not source_path.exists():
            if input_source == "clean":
                logger.error("Clean data not found: %s", source_path)
            else:
                logger.error("Oral data not found: %s", source_path)
            return 0

        source_data = read_jsonl(source_path)
        total = len(source_data)

        ckpt = Checkpoint(ckpt_path)
        processed_count = ckpt.get("processed_count", 0) if resume else 0

        remaining_items = source_data[processed_count:]

        if not remaining_items:
            logger.info("Scene '%s': all %d items already processed.", scene, total)
            return processed_count

        logger.info(
            "Scene '%s': mode=%s strength=%s source=%s total=%d processed=%d remaining=%d",
            scene, self.mode, self.strength, source_name, total, processed_count, len(remaining_items),
        )

        pbar = tqdm(total=len(remaining_items), desc=f"ASR-Sim[{scene}]", unit="item")

        for item in remaining_items:
            result = await self.simulate_one(scene, item)
            if result:
                append_jsonl([result], asr_path)
            processed_count += 1
            ckpt.set("processed_count", processed_count)
            pbar.update(1)

        pbar.close()
        return processed_count
