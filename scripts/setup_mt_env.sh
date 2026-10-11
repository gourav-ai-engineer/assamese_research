#!/usr/bin/env bash
# Separate environment for IndicTrans2 (needs transformers < 5). Reuses the system PyTorch.
# Usage: bash scripts/setup_mt_env.sh [ENV]  then  ENV/bin/python -m northeast_eval.cli mt-questions
set -euo pipefail
ENV=${1:-mtenv}
PKGS=("transformers>=4.44,<4.50" "huggingface_hub>=0.26,<1.0" sentencepiece IndicTransToolkit)
# Kaggle/Debian Pythons lack ensurepip, so create the env without pip and drive it with the
# system pip (`pip --python`, pip >= 22.3).
python -m venv --system-site-packages --without-pip "$ENV"
python -m pip --python "$ENV/bin/python" install -q "${PKGS[@]}"
"$ENV/bin/python" -c "import transformers, transformers.onnx; print('MT env transformers', transformers.__version__)"
