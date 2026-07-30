# -*- coding: utf-8 -*-
"""Lifecycle helpers for the local Gemma4/vLLM service.

The pipeline only needs local vLLM during LLM-backed steps. This module keeps
startup/shutdown decisions in the orchestrator layer instead of hiding them in
LLM clients or individual generation scripts.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import (  # noqa: E402
    ASR_STT_BASE_URL,
    ASR_TTS_BASE_URL,
    LLM_BASE_URL,
    USE_OPENROUTER,
    VLLM_HEALTH_INTERVAL,
    VLLM_HEALTH_TIMEOUT,
    VLLM_LOG_FILE,
    VLLM_PID_FILE,
    VLLM_START_SCRIPT,
    VLLM_STOP_TIMEOUT,
)

logger = logging.getLogger(__name__)


def _pid_path() -> Path:
    return Path(VLLM_PID_FILE)


def _log_path() -> Path:
    return Path(VLLM_LOG_FILE)


def _read_pid() -> int | None:
    path = _pid_path()
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        logger.warning("Invalid vLLM PID file, removing: %s", path)
        path.unlink(missing_ok=True)
        return None


def _write_pid(pid: int) -> None:
    path = _pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n", encoding="utf-8")


def _process_alive(pid: int) -> bool:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            parts = proc_stat.read_text(encoding="utf-8").split()
            if len(parts) > 2 and parts[2] == "Z":
                return False
        except OSError:
            pass
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _normalize_base_url(base_url: str | None = None) -> str:
    return (base_url or LLM_BASE_URL).rstrip("/")


def _health_url(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return base_url[:-3] + "/health"
    return base_url.rstrip("/") + "/health"


def _models_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url + "/models"
    return base_url + "/v1/models"


def probe_url(url: str, timeout: float = 5.0) -> bool:
    """Return True when an HTTP endpoint responds with 2xx."""
    try:
        response = httpx.get(url, timeout=timeout)
        return 200 <= response.status_code < 300
    except httpx.HTTPError:
        return False


def probe_vllm_ready(base_url: str | None = None, timeout: float = 5.0) -> bool:
    """Probe local vLLM readiness via /health, then /v1/models."""
    normalized = _normalize_base_url(base_url)
    return probe_url(_health_url(normalized), timeout=timeout) or probe_url(
        _models_url(normalized), timeout=timeout
    )


def wait_for_vllm_ready(base_url: str | None = None) -> None:
    """Wait until vLLM is ready or raise TimeoutError."""
    deadline = time.time() + float(VLLM_HEALTH_TIMEOUT)
    normalized = _normalize_base_url(base_url)
    while time.time() < deadline:
        if probe_vllm_ready(normalized):
            return
        time.sleep(float(VLLM_HEALTH_INTERVAL))
    raise TimeoutError(
        f"vLLM did not become ready within {VLLM_HEALTH_TIMEOUT}s. "
        f"Check log: {_log_path()}"
    )


def ensure_vllm_started(base_url: str | None = None) -> None:
    """Start local Gemma4/vLLM if needed and wait until ready.

    OpenRouter mode deliberately bypasses local lifecycle management.
    """
    if USE_OPENROUTER:
        logger.info("OpenRouter mode enabled; skip local vLLM startup.")
        return

    normalized = _normalize_base_url(base_url)
    pid = _read_pid()
    if pid and _process_alive(pid):
        if probe_vllm_ready(normalized):
            logger.info("Local vLLM already running (pid=%d).", pid)
            return
        logger.warning("vLLM pid=%d is alive but not ready; restarting.", pid)
        ensure_vllm_stopped()
    elif pid:
        logger.warning("Stale vLLM PID file found; removing: %s", _pid_path())
        _pid_path().unlink(missing_ok=True)

    if probe_vllm_ready(normalized):
        raise RuntimeError(
            "vLLM endpoint is already responding, but it was not started by this pipeline "
            f"(missing PID file: {_pid_path()}). Stop it manually or remove the conflict."
        )

    start_script = Path(VLLM_START_SCRIPT)
    if not start_script.exists():
        raise FileNotFoundError(f"vLLM start script not found: {start_script}")

    log_path = _log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Starting local Gemma4/vLLM via %s", start_script)
    logger.info("vLLM log: %s", log_path)

    log_file = open(log_path, "ab", buffering=0)
    try:
        proc = subprocess.Popen(
            ["bash", str(start_script)],
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log_file.close()
        raise

    _write_pid(proc.pid)
    try:
        wait_for_vllm_ready(normalized)
    except Exception:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM process exited during startup (exit={proc.returncode}). "
                f"Check log: {log_path}"
            )
        raise
    finally:
        log_file.close()

    logger.info("Local vLLM ready (pid=%d, base_url=%s).", proc.pid, normalized)


def ensure_vllm_stopped() -> None:
    """Stop the pipeline-managed local vLLM process, if any."""
    if USE_OPENROUTER:
        logger.info("OpenRouter mode enabled; skip local vLLM stop.")
        return

    pid = _read_pid()
    if pid is None:
        logger.info("No pipeline-managed vLLM PID file found; nothing to stop.")
        return

    if not _process_alive(pid):
        logger.info("vLLM pid=%d is not running; removing PID file.", pid)
        _pid_path().unlink(missing_ok=True)
        return

    logger.info("Stopping local Gemma4/vLLM (pid=%d).", pid)
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        _pid_path().unlink(missing_ok=True)
        return
    except PermissionError:
        logger.warning("No permission to stop vLLM pid=%d; leaving it running.", pid)
        return

    deadline = time.time() + float(VLLM_STOP_TIMEOUT)
    while time.time() < deadline:
        if not _process_alive(pid):
            _pid_path().unlink(missing_ok=True)
            logger.info("Local vLLM stopped.")
            return
        time.sleep(1)

    logger.warning("vLLM pid=%d did not exit after %ss; sending SIGKILL.", pid, VLLM_STOP_TIMEOUT)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _pid_path().unlink(missing_ok=True)
    logger.info("Local vLLM killed.")


def ensure_speech_services_ready(
    tts_base_url: str | None = None,
    stt_base_url: str | None = None,
) -> None:
    """Fail fast unless both local speech services respond to /health."""
    tts = (tts_base_url or ASR_TTS_BASE_URL).rstrip("/")
    stt = (stt_base_url or ASR_STT_BASE_URL).rstrip("/")
    missing = []
    for name, url in (("MOSS-TTS", f"{tts}/health"), ("Qwen3-ASR", f"{stt}/health")):
        if not probe_url(url, timeout=5.0):
            missing.append(f"{name} ({url})")
    if missing:
        raise RuntimeError(
            "Speech service health check failed: " + ", ".join(missing)
            + ". Start the TTS/ASR services before running --asr-mode tts_asr."
        )
    logger.info("Speech services ready: %s, %s", tts, stt)
