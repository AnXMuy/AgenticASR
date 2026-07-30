#!/bin/bash
# AgenticASR Gemma-4-31B-IT vLLM service launcher
# Usage: bash scripts/start_vllm.sh [options]

set -e

# Defaults
MODEL_PATH="${VLLM_MODEL_NAME:-google/gemma-4-31b-it}"
PORT=8000
HOST="0.0.0.0"
TENSOR_PARALLEL=1
MAX_MODEL_LEN=16384
GPU_MEMORY_UTILIZATION=0.9
DTYPE="bfloat16"
MAX_NUM_SEQS=24

# Parse options.
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --host)
            HOST="$2"; shift 2 ;;
        --tp)
            TENSOR_PARALLEL="$2"; shift 2 ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-util)
            GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --dtype)
            DTYPE="$2"; shift 2 ;;
        --max-num-seqs)
            MAX_NUM_SEQS="$2"; shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash scripts/start_vllm.sh [--model PATH] [--port PORT] [--tp N] [--max-model-len N] [--gpu-util F] [--dtype TYPE]"
            exit 1 ;;
    esac
done

# Environment checks.
echo "=========================================="
echo "  AgenticASR vLLM Server Launcher"
echo "=========================================="
echo "Model:          $MODEL_PATH"
echo "Port:           $PORT"
echo "Host:           $HOST"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "Max Model Len:  $MAX_MODEL_LEN"
echo "GPU Mem Util:   $GPU_MEMORY_UTILIZATION"
echo "Dtype:          $DTYPE"
echo "Max Num Seqs:   $MAX_NUM_SEQS"
echo "=========================================="

# Validate local paths while also allowing a Hugging Face model ID.
if [[ "$MODEL_PATH" == /* && ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[ERROR] Local model directory is missing config.json: $MODEL_PATH"
    exit 1
fi

# Check available GPUs.
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "Available GPUs: $GPU_COUNT"

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "[WARN] No GPU detected. Server will fail to load model."
    echo "[WARN] This script is intended for H100 environment."
fi

if [ "$GPU_COUNT" -gt 0 ] && [ "$TENSOR_PARALLEL" -gt "$GPU_COUNT" ]; then
    echo "[WARN] Requested TP=$TENSOR_PARALLEL but only $GPU_COUNT GPUs available. Adjusting to $GPU_COUNT."
    TENSOR_PARALLEL=$GPU_COUNT
fi

# Start the service.
echo ""
echo "Starting vLLM server..."
echo "API will be available at: http://${HOST}:${PORT}/v1"
echo "Press Ctrl+C to stop."
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
    --async-scheduling
