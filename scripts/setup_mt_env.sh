#!/usr/bin/env bash
# Separate environment for IndicTrans2 (needs transformers < 5). Reuses the system PyTorch.
# Usage: bash scripts/setup_mt_env.sh  then  mtenv/bin/python -m northeast_eval.cli mt-questions
set -euo pipefail
ENV=${1:-mtenv}
python -m venv --system-site-packages "$ENV"
"$ENV/bin/pip" install -q "transformers>=4.44,<4.50" sentencepiece IndicTransToolkit
"$ENV/bin/python" -c "import transformers; print('MT env transformers', transformers.__version__)"
