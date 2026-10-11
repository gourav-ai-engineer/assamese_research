#!/usr/bin/env bash
# Full pipeline on one GPU machine. Resumable: re-run after any interruption.
set -euo pipefail
OUT=${OUT:-results}
northeast-eval prepare
# IndicTrans2 needs transformers < 5, so MT steps run in their own environment.
bash scripts/setup_mt_env.sh mtenv
mtenv/bin/python -m northeast_eval.cli mt-questions
for m in gemma-4-E2B-it gemma-4-E4B-it gemma-3-1b-it gemma-3-4b-it Qwen3-1.7B Qwen3-4B Llama-3.2-1B Llama-3.2-3B sarvam-1; do
  northeast-eval run --model "$m" --out-dir "$OUT/predictions"
done
mtenv/bin/python -m northeast_eval.cli back-translate --pred-dir "$OUT/predictions"
mtenv/bin/python -m northeast_eval.cli mt-baseline --backend indictrans2 --out-dir "$OUT/predictions"
northeast-eval fertility --output "$OUT/fertility.csv"
northeast-eval score --pred-dir "$OUT/predictions" --output "$OUT/scores.csv"
northeast-eval report --scores "$OUT/scores.csv" --fertility-csv "$OUT/fertility.csv" --out-dir "$OUT"
northeast-eval sample-errors --pred-dir "$OUT/predictions" --output "$OUT/annotation/sample.csv"
