# Small Open LLMs on Northeast Indian Languages

Reproducible evaluation-only study of small open LLMs on Assamese (as), Manipuri/Meitei (mni), and Bodo (brx), with Hindi (hi) as a high-resource reference. No model training and no new dataset creation.

> Research integrity: this repository contains the experiment pipeline and templates, not completed experimental findings. Never report placeholder outputs as results. Verify exact benchmark configurations and model revisions before claiming comparability.

## Quick start (Python 3.11/3.12)
```bash
python -m venv .venv
# Windows: .\\.venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
python -m northeast_eval.cli validate --input data/examples/toy_translation.jsonl
python -m northeast_eval.cli score --predictions data/examples/toy_predictions.jsonl --output results/toy_metrics.json
pytest
```

## Research questions
- Compare small open models on English↔Assamese, English↔Manipuri and English↔Bodo translation.
- Compare zero-shot, same-language one-shot and Hindi one-shot prompts.
- Analyze tokenizer fertility and wrong-script/language errors.

Use [IndicGenBench](https://github.com/AI4Bharat/IndicGenBench) and its official licensing/access instructions. Verify dataset language coverage before use; do not assume all resources contain all three languages. Do not commit licensed test data unless permitted. Planned candidates: Gemma 3 1B/4B IT, Qwen3 1.7B/4B, Llama 3.2 1B/3B Instruct and Sarvam-1; verify revisions, licenses and language coverage. See `paper/PROTOCOL.md` for protocol and reporting guardrails. No benchmark results are included yet; two weeks is a sprint target, not a publication guarantee.