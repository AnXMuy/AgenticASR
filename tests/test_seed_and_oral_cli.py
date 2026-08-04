import asyncio
import importlib.util
import json
import logging
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "pipeline" / "scripts" / "01_seed_and_oral.py"


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


class DummyAsyncLLMClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyOralTextGenerator:
    calls = []

    def __init__(self, async_client=None):
        self.async_client = async_client

    async def generate_scene(self, scene, total_num, resume=True):
        self.calls.append((scene, total_num, resume))
        return total_num


class DummySeedGenerator:
    def __init__(self, async_client=None):
        self.async_client = async_client

    async def generate_incremental(self, scene, round_index):
        return {}


class DummyPromptLoader:
    def list_scenes(self, step):
        return ["daily_chat"]


def load_script():
    settings = types.ModuleType("configs.settings")
    settings.RAW_DIR = "/unused/raw"
    settings.ROUND_SIZE = 100
    settings.SCENE_DISTRIBUTION = {"daily_chat": 0.07}
    settings.SEEDS_DIR = "/unused/seeds"
    settings.SEED_TARGET_COUNT = 300
    settings.SEED_PER_ROUND = 100
    settings.TOTAL_SAMPLES = 100_000

    io_utils = types.ModuleType("src.utils.io_utils")
    io_utils.read_jsonl = _read_jsonl
    io_utils.setup_logging = lambda *args, **kwargs: None
    io_utils.write_json = lambda data, path: Path(path).write_text(
        json.dumps(data), encoding="utf-8"
    )

    modules = {
        "configs": types.ModuleType("configs"),
        "configs.settings": settings,
        "src": types.ModuleType("src"),
        "src.generators": types.ModuleType("src.generators"),
        "src.generators.oral_text_generator": types.ModuleType(
            "src.generators.oral_text_generator"
        ),
        "src.generators.seed_generator": types.ModuleType("src.generators.seed_generator"),
        "src.services": types.ModuleType("src.services"),
        "src.services.llm_client": types.ModuleType("src.services.llm_client"),
        "src.utils": types.ModuleType("src.utils"),
        "src.utils.io_utils": io_utils,
        "src.utils.prompt_loader": types.ModuleType("src.utils.prompt_loader"),
    }
    modules["src.generators.oral_text_generator"].OralTextGenerator = (
        DummyOralTextGenerator
    )
    modules["src.generators.seed_generator"].SeedGenerator = DummySeedGenerator
    modules["src.generators.seed_generator"].dedupe_seed_dict = lambda seeds: seeds
    modules["src.services.llm_client"].AsyncLLMClient = DummyAsyncLLMClient
    modules["src.utils.prompt_loader"].PromptLoader = DummyPromptLoader

    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("seed_and_oral_script", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


class SeedAndOralCliTest(unittest.TestCase):
    def setUp(self):
        DummyOralTextGenerator.calls = []
        self.module = load_script()

    def test_rounds_sets_incremental_generation_target(self):
        with TemporaryDirectory() as tmp:
            self.module.RAW_DIR = tmp
            scene_dir = Path(tmp) / "daily_chat"
            scene_dir.mkdir(parents=True)
            oral_path = scene_dir / "oral.jsonl"
            oral_path.write_text(
                "\n".join(json.dumps({"id": i}) for i in range(3)) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                self.module._oral_generation_target(
                    "daily_chat", distribution_target=7_000, rounds=2, resume=True
                ),
                (3, 203, 2),
            )

    def test_no_resume_rounds_starts_from_zero(self):
        with TemporaryDirectory() as tmp:
            self.module.RAW_DIR = tmp
            scene_dir = Path(tmp) / "daily_chat"
            scene_dir.mkdir(parents=True)
            (scene_dir / "oral.jsonl").write_text('{"id": 1}\n', encoding="utf-8")

            self.assertEqual(
                self.module._oral_generation_target(
                    "daily_chat", distribution_target=7_000, rounds=1, resume=False
                ),
                (0, 100, 1),
            )

    def test_phase2_passes_resume_flag_to_generator(self):
        with TemporaryDirectory() as tmp:
            self.module.RAW_DIR = tmp

            asyncio.run(
                self.module.phase2_oral(
                    ["daily_chat"],
                    {"daily_chat": 123},
                    None,
                    False,
                    logging.getLogger(__name__),
                )
            )

        self.assertEqual(DummyOralTextGenerator.calls, [("daily_chat", 123, False)])


if __name__ == "__main__":
    unittest.main()
