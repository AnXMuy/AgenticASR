"""Generate scene-conditioned Oral utterances from candidate seed pools.

The implementation follows the paper's independent controls: one of four
correction structures and one of three colloquialization levels. Explanation
cues are enabled only for the explanation scene. Passthrough samples disable
both controls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.scenes import SCENE_REGISTRY
from configs.settings import LLM_MAX_CONCURRENT, LLM_MAX_TOKENS_ORAL, PASSTHROUGH_RATIO
from src.services.llm_client import AsyncLLMClient, LLMClient
from src.utils.io_utils import Checkpoint, append_jsonl

from tqdm import tqdm

logger = logging.getLogger(__name__)

CORRECTION_STRUCTURES = {
    "none": (
        "Do not introduce a self-correction. Preserve one consistent intended meaning "
        "throughout the utterance."
    ),
    "single": (
        "Introduce exactly one explicit correction from A to B. The final intended "
        "value must be B."
    ),
    "rollback": (
        "Introduce a rollback correction: say A, change it to B, then explicitly return "
        "to A. The final intended value must be A."
    ),
    "multiple": (
        "Introduce a multi-stage correction: say A, revise it to B, and later revise it "
        "to C. The final intended value must be C."
    ),
}

ORAL_LEVELS = {
    "low": (
        "Use clean conversational speech with at most one mild filler or repetition."
    ),
    "moderate": (
        "Use natural spontaneous speech with several dispersed fillers, hesitations, "
        "or content-word repetitions."
    ),
    "high": (
        "Use strongly colloquial but still understandable speech with varied fillers, "
        "false starts, stuttering, and content-word repetitions."
    ),
}

SCENE_CONTEXTS = {
    "daily_chat": "an informal everyday conversation",
    "english_daily": "an informal everyday conversation",
    "vibe_coding": "a spoken request to an AI coding assistant",
    "explanation": "a name, place, or uncommon entity clarification",
    "meeting": "an impromptu workplace meeting contribution",
    "english_meeting": "an impromptu workplace meeting contribution",
    "customer_service": "a customer speaking to a support representative",
    "english_customer_service": "a customer speaking to a support representative",
    "academic": "a research or education discussion",
    "english_academic": "a research or education discussion",
    "navigation": "a navigation or travel request",
    "dictation_memo": "a dictated note or task list",
    "english_dictation": "a dictated note or task list",
    "voice_search": "a command to a voice assistant",
    "english_voice_search": "a command to a voice assistant",
    "english_tech": "a spoken technical or DevOps instruction",
}

SYSTEM_PROMPT = """You generate one realistic Oral transcript for Agentic Speech Recognition.
The transcript must sound spoken rather than written, remain coherent for the requested scene,
and preserve one unambiguous final intended meaning. Use three to five of the five candidate
seeds. Follow the requested language, correction structure, and oral level exactly. Do not add
facts that conflict with the selected seeds. Return strict JSON only:
{"oral_text":"...","selected_seeds":["..."],"speech_phenomena":["..."]}."""

USER_TEMPLATE = """Scene: {scene}
Situation: {context}
Output language: {language}
Five candidate seeds: {seeds}

Correction control:
{correction}

Oral control:
{oral_level}

Additional control:
{additional}

Requirements:
- Select and use three to five candidate seeds.
- Write one utterance, not a dialogue or explanation of the task.
- Keep numbers and dates in natural spoken form unless this is a passthrough sample.
- Keep technical terms in their standard language; do not create phonetic transliterations.
- Use only ordinary sentence punctuation and no Markdown.
- Make the speaker's final intended content recoverable without outside knowledge."""

EXPLANATION_CONTROL = (
    "Include a concise spoken spelling, word-composition, or contrastive explanation for "
    "one uncommon or ambiguous entity. The final spelling must be recoverable from the cue."
)

PASSTHROUGH_CONTROL = (
    "This is a passthrough sample. Disable correction and oral controls. Produce already clean "
    "written-form text with no fillers, repetitions, false starts, or spoken-form numbers."
)


def _flatten_seed_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_seed_values(item))
        return flattened
    if isinstance(value, dict):
        preferred = (
            value.get("term")
            or value.get("name")
            or value.get("keyword")
            or value.get("entity")
        )
        if isinstance(preferred, str) and preferred.strip():
            return [preferred.strip()]
        flattened = []
        for nested in value.values():
            flattened.extend(_flatten_seed_values(nested))
        return flattened
    return []


class SeedSampler:
    """Load, shuffle, and cycle through a scene's seed pool."""

    def __init__(self, seeds_dir: Path, rng: random.Random | None = None) -> None:
        self.seeds_dir = seeds_dir
        self.rng = rng or random.Random()
        self._pools: dict[str, list[str]] = {}
        self._offsets: dict[str, int] = {}

    def sample(self, scene: str, count: int = 5) -> list[str]:
        if count < 1:
            raise ValueError("count must be at least 1")
        pool = self._pools.get(scene)
        if pool is None:
            pool = self._load(scene)
            self._pools[scene] = pool
            self._offsets[scene] = 0
        if len(pool) < count:
            raise ValueError(f"scene {scene!r} has only {len(pool)} usable seeds")

        offset = self._offsets[scene]
        if offset + count > len(pool):
            self.rng.shuffle(pool)
            offset = 0
        selected = pool[offset : offset + count]
        self._offsets[scene] = offset + count
        return selected

    def _load(self, scene: str) -> list[str]:
        path = self.seeds_dir / f"{scene}.json"
        if not path.exists():
            raise FileNotFoundError(f"seed file not found: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        seeds = _flatten_seed_values(value)
        unique = list(dict.fromkeys(seeds))
        self.rng.shuffle(unique)
        return unique


class OralTextGenerator:
    """Generate Oral records asynchronously with resumable JSONL output."""

    def __init__(
        self,
        async_client: AsyncLLMClient | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.llm = async_client or AsyncLLMClient(max_concurrent=LLM_MAX_CONCURRENT)
        self.rng = rng or random.Random()
        self.seeds_dir = PROJECT_ROOT / "data" / "seeds"
        self.raw_dir = PROJECT_ROOT / "data" / "raw"
        self.sampler = SeedSampler(self.seeds_dir, self.rng)

    def _language(self, scene: str) -> str:
        configured = SCENE_REGISTRY.get(scene, {}).get("language", "zh")
        if configured == "en":
            return "English"
        if configured == "zh-en-mix":
            return "Chinese with natural English technical terms when needed"
        return "Chinese"

    def _controls(self, scene: str) -> tuple[str, str, str, list[str]]:
        if self.rng.random() < PASSTHROUGH_RATIO:
            return PASSTHROUGH_CONTROL, PASSTHROUGH_CONTROL, "None", ["clean"]

        correction_name = self.rng.choice(tuple(CORRECTION_STRUCTURES))
        oral_name = self.rng.choice(tuple(ORAL_LEVELS))
        additional = EXPLANATION_CONTROL if scene == "explanation" else "None"
        phenomena = [f"correction:{correction_name}", f"oral:{oral_name}"]
        if scene == "explanation":
            phenomena.append("explanation")
        return (
            CORRECTION_STRUCTURES[correction_name],
            ORAL_LEVELS[oral_name],
            additional,
            phenomena,
        )

    async def generate_one(self, scene: str, item_id: int) -> dict[str, object] | None:
        candidates = self.sampler.sample(scene, count=5)
        correction, oral_level, additional, sampled_phenomena = self._controls(scene)
        user_prompt = USER_TEMPLATE.format(
            scene=scene,
            context=SCENE_CONTEXTS.get(scene, "a realistic voice interaction"),
            language=self._language(scene),
            seeds=json.dumps(candidates, ensure_ascii=False),
            correction=correction,
            oral_level=oral_level,
            additional=additional,
        )
        try:
            response = await self.llm.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self.rng.uniform(0.8, 0.95),
                max_tokens=LLM_MAX_TOKENS_ORAL,
            )
            parsed = LLMClient._parse_json(response)
            if not isinstance(parsed, dict):
                raise ValueError("Oral generator response must be a JSON object")
            oral_text = parsed.get("oral_text")
            if not isinstance(oral_text, str) or len(oral_text.strip()) < 4:
                raise ValueError("Oral generator returned empty or too-short text")
            selected = parsed.get("selected_seeds")
            if not isinstance(selected, list):
                selected = candidates
            returned_phenomena = parsed.get("speech_phenomena")
            if sampled_phenomena == ["clean"]:
                phenomena = ["clean"]
            elif isinstance(returned_phenomena, list):
                phenomena = [str(item) for item in returned_phenomena]
            else:
                phenomena = sampled_phenomena
            return {
                "id": item_id,
                "oral_text": oral_text.strip(),
                "selected_seeds": selected,
                "speech_phenomena": phenomena,
                "scene": scene,
            }
        except Exception as error:
            logger.warning("Oral generation failed for id=%s: %s", item_id, error)
            return None

    async def generate_scene(
        self,
        scene: str,
        total_num: int,
        resume: bool = True,
    ) -> int:
        output_dir = self.raw_dir / scene
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "oral.jsonl"
        checkpoint = Checkpoint(output_dir / "oral.ckpt.json")
        generated = checkpoint.get("generated_count", 0) if resume else 0
        attempted = checkpoint.get("attempted_count", generated) if resume else 0
        if not resume and output_path.exists():
            output_path.unlink()

        progress = tqdm(total=max(0, total_num - generated), desc=f"Oral[{scene}]", unit="item")
        while generated < total_num:
            batch_size = min(LLM_MAX_CONCURRENT * 2, total_num - generated)
            results = await asyncio.gather(
                *(self.generate_one(scene, attempted + offset) for offset in range(batch_size))
            )
            attempted += batch_size
            valid = [record for record in results if record is not None]
            if not valid:
                raise RuntimeError(f"all Oral generation requests failed for scene {scene}")
            append_jsonl(valid, output_path)
            generated += len(valid)
            checkpoint.set("generated_count", generated)
            checkpoint.set("attempted_count", attempted)
            progress.update(len(valid))
        progress.close()
        return generated
