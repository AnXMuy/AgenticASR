# -*- coding: utf-8 -*-
"""HTTP wrapper for a local Qwen3-ASR model server."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Qwen3ASRService:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from qwen3_asr import Qwen3ASR  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-ASR dependency is not installed. Install the project/package "
                "that provides `qwen3_asr.Qwen3ASR`, or adapt Qwen3ASRService._load_model() "
                "to your local Qwen3-ASR entrypoint."
            ) from exc
        self._model = Qwen3ASR.from_pretrained(self.model_path, device=self.device)
        return self._model

    def transcribe(self, audio_path: Path, language: str | None = None) -> str:
        if not audio_path.exists():
            raise RuntimeError(f"Audio file not found: {audio_path}")
        model = self._load_model()
        kwargs: dict[str, Any] = {"audio_path": str(audio_path)}
        if language:
            kwargs["language"] = language
        if hasattr(model, "transcribe"):
            result = model.transcribe(**kwargs)
        elif hasattr(model, "asr"):
            result = model.asr(**kwargs)
        else:
            raise RuntimeError("Qwen3-ASR model has no transcribe() or asr() method.")
        if isinstance(result, dict):
            return str(result.get("text") or result.get("transcript") or "")
        return str(result or "")


class ASRRequest(BaseModel):
    audio_path: str = Field(min_length=1)
    language: str | None = None


def create_app(model_path: str, device: str = "auto"):
    try:
        from fastapi import Body, FastAPI
    except ImportError as exc:
        raise RuntimeError("Install fastapi, pydantic and uvicorn to run the Qwen3-ASR service.") from exc

    service = Qwen3ASRService(model_path=model_path, device=device)
    app = FastAPI(title="AgenticASR Qwen3-ASR Service")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/asr")
    def transcribe(request: ASRRequest = Body(...)) -> dict[str, str]:
        text = service.transcribe(audio_path=Path(request.audio_path), language=request.language).strip()
        return {"text": text}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start local Qwen3-ASR HTTP service")
    parser.add_argument("--model-path", default=os.environ.get("QWEN3_ASR_MODEL_PATH", ""), help="Qwen3-ASR model path")
    parser.add_argument("--device", default=os.environ.get("ASR_DEVICE", "auto"), help="auto/cpu/cuda")
    parser.add_argument("--host", default=os.environ.get("QWEN3_ASR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("QWEN3_ASR_PORT", "8012")))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    if not args.model_path:
        raise SystemExit("--model-path or QWEN3_ASR_MODEL_PATH is required")

    import uvicorn

    app = create_app(model_path=args.model_path, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
