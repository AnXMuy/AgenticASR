#!/usr/bin/env bash

set -euo pipefail

MODEL_DIR="${1:-models}"
VAD_PATH="${MODEL_DIR}/silero_vad.onnx"
VAD_URL="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

mkdir -p "${MODEL_DIR}"
if [[ -f "${VAD_PATH}" ]]; then
    printf 'Already present: %s\n' "${VAD_PATH}"
    exit 0
fi

curl --fail --location --output "${VAD_PATH}" "${VAD_URL}"
printf 'Downloaded: %s\n' "${VAD_PATH}"
