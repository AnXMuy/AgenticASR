# -*- coding: utf-8 -*-
"""HTTP wrapper for a local MOSS-TTS model server.

Run this service separately from the data pipeline so Step 3 and benchmark tools
can call it via HTTP without loading speech models in their own processes.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


class MossTTSService:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch_device: Any | None = None
        self._sample_rate: int | None = None

    def _load_model(self) -> tuple[Any, Any | None, Any | None, int | None]:
        if self._model is not None:
            return self._model, self._processor, self._torch_device, self._sample_rate

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            # Backward-compatible fallback for older/local wrappers.
            try:
                from moss_tts import MossTTS  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "MOSS-TTS dependencies are not installed. Install transformers/torch with "
                    "MOSS-TTS trust_remote_code support, or install a package that provides "
                    "`moss_tts.MossTTS`."
                ) from exc
            self._model = MossTTS.from_pretrained(self.model_path, device=self.device)
            return self._model, None, None, None

        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        device_str = self.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        torch_device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
        dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32
        attn_implementation = _resolve_attn_implementation(torch_device, dtype)

        logger.info("Loading MOSS-TTS model=%s device=%s attn=%s", self.model_path, torch_device, attn_implementation)
        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        if hasattr(processor, "audio_tokenizer") and processor.audio_tokenizer is not None:
            processor.audio_tokenizer = processor.audio_tokenizer.to(torch_device)

        model_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": dtype}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModel.from_pretrained(self.model_path, **model_kwargs).to(torch_device)
        model.eval()

        self._model = model
        self._processor = processor
        self._torch_device = torch_device
        self._sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
        return self._model, self._processor, self._torch_device, self._sample_rate

    def synthesize(
        self,
        text: str,
        output_path: Path,
        language: str | None = None,
        strength: str = "medium",
        mode: str = "direct",
        reference_audio: str | None = None,
        reference_transcript: str | None = None,
        tokens: int | None = None,
        max_new_tokens: int = 4096,
        audio_temperature: float = 1.7,
        audio_top_p: float = 0.8,
        audio_top_k: int = 25,
        audio_repetition_penalty: float = 1.0,
    ) -> dict[str, Any]:
        model, processor, torch_device, sample_rate = self._load_model()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if processor is None:
            path = self._synthesize_with_legacy_wrapper(
                model=model,
                text=text,
                output_path=output_path,
                language=language,
                strength=strength,
            )
            return {"audio_path": str(path), "sample_rate": sample_rate, "backend": "legacy_wrapper"}

        path = self._synthesize_with_transformers(
            model=model,
            processor=processor,
            torch_device=torch_device,
            text=text,
            output_path=output_path,
            language=language,
            mode=mode,
            reference_audio=reference_audio,
            reference_transcript=reference_transcript,
            tokens=tokens,
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_p=audio_top_p,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
        )
        return {
            "audio_path": str(path),
            "sample_rate": sample_rate,
            "backend": "transformers",
            "mode_used": mode,
            "reference_audio_used": reference_audio,
            "tokens_used": tokens,
            "generation_params": {
                "max_new_tokens": max_new_tokens,
                "audio_temperature": audio_temperature,
                "audio_top_p": audio_top_p,
                "audio_top_k": audio_top_k,
                "audio_repetition_penalty": audio_repetition_penalty,
            },
        }

    def _synthesize_with_legacy_wrapper(
        self,
        model: Any,
        text: str,
        output_path: Path,
        language: str | None,
        strength: str,
    ) -> Path:
        kwargs: dict[str, Any] = {"text": text, "output_path": str(output_path)}
        if language:
            kwargs["language"] = language
        if strength:
            kwargs["strength"] = strength
        if hasattr(model, "synthesize_to_file"):
            model.synthesize_to_file(**kwargs)
        elif hasattr(model, "tts_to_file"):
            model.tts_to_file(**kwargs)
        else:
            raise RuntimeError("MOSS-TTS model has no synthesize_to_file() or tts_to_file() method.")
        if not output_path.exists():
            raise RuntimeError(f"MOSS-TTS did not create audio file: {output_path}")
        return output_path

    def _synthesize_with_transformers(
        self,
        *,
        model: Any,
        processor: Any,
        torch_device: Any,
        text: str,
        output_path: Path,
        language: str | None,
        mode: str,
        reference_audio: str | None,
        reference_transcript: str | None,
        tokens: int | None,
        max_new_tokens: int,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        audio_repetition_penalty: float,
    ) -> Path:
        import torch
        import torchaudio

        conversations, processor_mode = _build_conversations(
            processor=processor,
            text=text,
            language=language,
            mode=mode,
            reference_audio=reference_audio,
            reference_transcript=reference_transcript,
            tokens=tokens,
        )
        batch = processor(conversations, mode=processor_mode)
        input_ids = batch["input_ids"].to(torch_device)
        attention_mask = batch["attention_mask"].to(torch_device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                audio_temperature=audio_temperature,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
            )
        decoded = processor.decode(outputs)
        if not decoded or not decoded[0].audio_codes_list:
            raise RuntimeError("MOSS-TTS returned no audio codes.")
        audio = decoded[0].audio_codes_list[0]
        sample_rate = int(getattr(processor.model_config, "sampling_rate", 24000))
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), sample_rate)
        if not output_path.exists():
            raise RuntimeError(f"MOSS-TTS did not create audio file: {output_path}")
        return output_path


def _build_conversations(
    *,
    processor: Any,
    text: str,
    language: str | None,
    mode: str,
    reference_audio: str | None,
    reference_transcript: str | None,
    tokens: int | None,
) -> tuple[list[list[Any]], str]:
    user_kwargs: dict[str, Any] = {"text": text}
    if language:
        user_kwargs["language"] = language
    if tokens is not None:
        user_kwargs["tokens"] = tokens

    if mode == "direct":
        return [[processor.build_user_message(**user_kwargs)]], "generation"

    if mode == "clone":
        if not reference_audio:
            raise ValueError("mode=clone requires reference_audio")
        user_kwargs["reference"] = [reference_audio]
        return [[processor.build_user_message(**user_kwargs)]], "generation"

    if mode == "continuation":
        if not reference_audio or not reference_transcript:
            raise ValueError("mode=continuation requires reference_audio and reference_transcript")
        cont_kwargs = dict(user_kwargs)
        cont_kwargs["text"] = reference_transcript + text
        return [[
            processor.build_user_message(**cont_kwargs),
            processor.build_assistant_message(audio_codes_list=[reference_audio]),
        ]], "continuation"

    if mode == "continuation_clone":
        if not reference_audio or not reference_transcript:
            raise ValueError("mode=continuation_clone requires reference_audio and reference_transcript")
        cont_kwargs = dict(user_kwargs)
        cont_kwargs["text"] = reference_transcript + text
        cont_kwargs["reference"] = [reference_audio]
        return [[
            processor.build_user_message(**cont_kwargs),
            processor.build_assistant_message(audio_codes_list=[reference_audio]),
        ]], "continuation"

    raise ValueError(f"Unsupported MOSS-TTS mode: {mode}")


def _resolve_attn_implementation(device: Any, dtype: Any) -> str:
    try:
        import torch
    except ImportError:
        return "eager"
    if (
        str(device).startswith("cuda")
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if str(device).startswith("cuda"):
        return "sdpa"
    return "eager"


def create_app(model_path: str, device: str = "auto"):
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("Install fastapi, pydantic and uvicorn to run the MOSS-TTS service.") from exc

    service = MossTTSService(model_path=model_path, device=device)
    app = FastAPI(title="AgenticASR MOSS-TTS Service")

    class TTSRequest(BaseModel):
        text: str = Field(min_length=1)
        output_path: str
        language: str | None = None
        strength: str = "medium"
        mode: Literal["direct", "clone", "continuation", "continuation_clone"] = "direct"
        reference_audio: str | None = None
        reference_transcript: str | None = None
        tokens: int | None = None
        max_new_tokens: int = 4096
        audio_temperature: float = 1.7
        audio_top_p: float = 0.8
        audio_top_k: int = 25
        audio_repetition_penalty: float = 1.0

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/tts")
    def synthesize(request: TTSRequest) -> dict[str, Any]:
        return service.synthesize(
            text=request.text,
            output_path=Path(request.output_path),
            language=request.language,
            strength=request.strength,
            mode=request.mode,
            reference_audio=request.reference_audio,
            reference_transcript=request.reference_transcript,
            tokens=request.tokens,
            max_new_tokens=request.max_new_tokens,
            audio_temperature=request.audio_temperature,
            audio_top_p=request.audio_top_p,
            audio_top_k=request.audio_top_k,
            audio_repetition_penalty=request.audio_repetition_penalty,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start local MOSS-TTS HTTP service")
    parser.add_argument("--model-path", default=os.environ.get("MOSS_TTS_MODEL_PATH", ""), help="MOSS-TTS model path")
    parser.add_argument("--device", default=os.environ.get("ASR_DEVICE", "auto"), help="auto/cpu/cuda")
    parser.add_argument("--host", default=os.environ.get("MOSS_TTS_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MOSS_TTS_PORT", "8011")))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    if not args.model_path:
        raise SystemExit("--model-path or MOSS_TTS_MODEL_PATH is required")

    import uvicorn

    app = create_app(model_path=args.model_path, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
