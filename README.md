# Small Open LLMs on Northeast Indian Languages

Evaluation-only study: **How Well Do Small Open LLMs Handle Northeast Indian Languages? An Empirical Study on Assamese, Manipuri and Bodo.** No training, no new dataset. Seven 1–4B open models are run on existing IndicGenBench test sets, with Hindi as a high-resource reference.

> **Research integrity.** The repository holds the pipeline, not results. Never report smoke-test or placeholder outputs as findings. Pin model revisions before the final runs.

## What it does

| Stage | Command | Output |
|---|---|---|
| Data | `northeast-eval prepare` | Downloads IndicGenBench; writes frozen samples to `data/processed/{eval,exemplars}/` |
| Inference | `northeast-eval run [--model X]` | `results/predictions/<model>/<file>__<condition>.jsonl` (resumable) |
| MT reference | `northeast-eval mt-baseline` | IndicTrans2 on the same FLORES items |
| Tokenizers | `northeast-eval fertility` | `results/fertility.csv` (tokens/word per model × language) |
| Scoring | `northeast-eval score` | `results/scores.csv`: chrF / token-F1, 95% bootstrap CI, wrong-script/empty/copy rates |
| Report | `northeast-eval report` | `results/tables/*.md,*.tex`, `results/figures/*.pdf,png` |
| Error analysis | `northeast-eval sample-errors` | Blind 50-per-language annotation sheet + separate key |
| Script study (optional) | `northeast-eval prepare-in22` | Meitei-script Manipuri from IN22-Gen |

`scripts/run_all.sh` runs every stage; `notebooks/colab_run.ipynb` does the same on a free Colab T4, saving to Google Drive.

## Design

- **Languages:** Assamese (`as`, Bengali script), Manipuri (`mni`; Bengali script in IndicGenBench), Bodo (`brx`, Devanagari), Hindi (`hi`, reference).
- **Tasks:** FLORES-IN translation, both directions (chrF); XorQA-IN cross-lingual QA (max token-F1 over gold answers). CrossSum-IN summarization can be switched on in `configs/experiment.yaml`.
- **Models** (`configs/models.yaml`): Gemma 3 1B/4B-IT, Qwen3 1.7B/4B (thinking disabled), Llama 3.2 1B/3B-Instruct, Sarvam-1 (base model, few-shot only).
- **Conditions:** `zero_shot`, `few_shot` (3 same-language exemplars from dev/train), `few_shot_hindi` (Hindi exemplars), `translate_test` (QA: question → English via IndicTrans2, answer in English, translate back).
- **Decoding:** greedy; translation 160 new tokens, QA 32.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # pipeline, scoring, tests (no GPU needed)
pytest
northeast-eval prepare             # 500 sentences/direction, 300 QA items/language
pip install -e ".[gpu,mt]"         # on the GPU machine
huggingface-cli login              # Gemma, Llama and IndicTrans2 are gated
northeast-eval run --model gemma-3-1b-it --limit 10   # smoke test
./scripts/run_all.sh
```

## Layout

```
configs/            experiment grid and model list
northeast_eval/     data, prompts, generation, translator, metrics, fertility, reporting, runner, cli
notebooks/          Colab runner
paper/              LaTeX draft (main.tex), refs.bib, protocol, outline, reference checklist
scripts/run_all.sh  end-to-end pipeline
tests/              unit tests and a fake-model end-to-end test
```

## Data licences

IndicGenBench: FLORES-IN CC BY-SA 4.0, XorQA-IN MIT, CrossSum-IN CC BY-NC-SA 4.0. Data is downloaded at run time and not committed (`data/raw`, `data/processed`, `results` are git-ignored). See `paper/PROTOCOL.md` for the frozen protocol and reporting checklist.
