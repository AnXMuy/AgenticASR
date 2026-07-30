# Experiments and Inference

The public scripts here cover inference and benchmark scoring. Paper-only plots, ablations, and result archives are not included.

## Refiner Inference

```bash
python experiments/scripts/postprocess_asr.py \
  path/to/asr_output.jsonl \
  path/to/refined_output.jsonl \
  --model /path/to/refiner-checkpoint
```

## Benchmark Judge

Download `rubric.json` from the [ModelScope](https://www.modelscope.cn/datasets/MuyuanJ/AASR-Bench) or [Hugging Face](https://huggingface.co/datasets/Andrew0425/AASR-Bench) release, then run:

```bash
python experiments/scripts/main.py \
  path/to/system_output.jsonl \
  --rubric /path/to/rubric.json \
  --results path/to/judge_results.jsonl \
  --summary path/to/judge_summary.json \
  --api-url http://127.0.0.1:8000/v1 \
  --model /path/to/judge-model
```

See `scripts/README.md` for the file map.
