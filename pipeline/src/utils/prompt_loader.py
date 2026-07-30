"""Load seed, Oral, and Clean prompt configuration."""

from __future__ import annotations

import json
import random
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
PROMPTS_SEED_PATH = CONFIGS_DIR / "prompts.json"
PROMPTS_CLEAN_PATH = CONFIGS_DIR / "prompts_clean.json"
PROMPTS_ORAL_PATH = CONFIGS_DIR / "prompts_oral.json"


class PromptLoader:
    """Load and render prompt templates from JSON configuration."""

    _STYLES = (
        "A short declarative utterance with one simple intent.",
        "A short question or command with one simple intent.",
        "A medium utterance containing two or three information units.",
        "A medium question containing two or three information units.",
        "A long declarative utterance containing several concrete details.",
        "A long question or command containing several concrete details.",
    )

    def __init__(self) -> None:
        self._seed_config = self._load_json(PROMPTS_SEED_PATH)
        self._clean_config = self._load_json(PROMPTS_CLEAN_PATH)
        self._oral_config = self._load_json(PROMPTS_ORAL_PATH)

    @staticmethod
    def _load_json(path: Path) -> dict:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def get_seed_prompt(self, scene: str) -> dict[str, str]:
        """Return the system and user prompts for one seed scene."""

        config = self._seed_config["seed_generation"][scene]
        return {"system": config["system"], "user": config["user_template"]}

    @classmethod
    def sample_style(cls) -> str:
        return random.choice(cls._STYLES)

    def get_clean_prompt(
        self,
        scene: str = "zh",
        seeds: str = "",
        style: str = "",
    ) -> dict[str, str]:
        """Return a Clean-generation prompt for a language key."""

        del seeds, style
        return self.get_clean_prompt_by_lang(scene)

    def get_clean_prompt_by_lang(self, lang: str = "zh") -> dict[str, str]:
        """Return the configured Clean prompt, falling back to Chinese."""

        systems = self._clean_config.get("system_prompts", {})
        users = self._clean_config.get("user_template", {})
        return {
            "system": systems.get(lang, systems.get("zh", "")),
            "user": users.get(lang, users.get("zh", "")),
        }

    def list_scenes(self, step: str = "clean") -> list[str]:
        """List configured scene or language keys for a pipeline stage."""

        if step == "seed":
            config = self._seed_config.get("seed_generation", {})
        elif step == "oral":
            config = self._oral_config.get("system_prompts", {})
        elif step == "clean":
            config = self._clean_config.get("system_prompts", {})
        else:
            config = {}
        return [key for key in config if not key.startswith("_")]

    def format_seeds_for_prompt(self, seeds: dict | list, max_total: int = 30) -> str:
        """Sample and format a compact list from a seed pool."""

        items: list[str] = []
        if isinstance(seeds, dict):
            values = list(seeds.values())
        elif isinstance(seeds, list):
            values = seeds
        else:
            return str(seeds)
        for value in values:
            entries = value if isinstance(value, list) else [value]
            for entry in entries:
                if isinstance(entry, dict):
                    name = next(
                        (entry.get(key) for key in ("name", "term", "keyword", "param") if entry.get(key)),
                        json.dumps(entry, ensure_ascii=False, sort_keys=True),
                    )
                    items.append(str(name))
                elif isinstance(entry, str):
                    items.append(entry)
        if not items:
            return "Use appropriate scene-specific content."
        unique = list(dict.fromkeys(items))
        random.shuffle(unique)
        return ", ".join(unique[:max_total])

    def format_exclusion_list(self, patterns: list[str], max_items: int = 10) -> str:
        """Format previously used patterns for an exclusion prompt."""

        if not patterns:
            return "None"
        return ", ".join(f'"{pattern}"' for pattern in patterns[:max_items])
