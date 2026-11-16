#!/bin/bash
# Run once on the login node before submitting jobs (no GPU needed):
#   1. download the models into the shared HF cache; data/model_ready/<key> marks each one done;
#   2. download and normalise the six public tables into data/real/ (code/prepare_real.py);
#   3. build every task set in config.json (v1, v1real, v2, v2real); task sets that already
#      exist are kept as they are;
#   4. run the CPU self-test (scripted agent, 6 tasks, all four memory conditions) on each task set.
#
#   bash download.sh            # every model in config.json
#   bash download.sh q07 q14    # selected models
#
# Llama-3.1-8B-Instruct (l08) is gated: it needs a token in $USER_STORAGE/.hf_token and an account
# approved by Meta. If the download fails, l08 is not marked and submit-all.sh skips it.
# If the machine cannot reach UCI, copy the original files into data/real/raw/ by hand (file names:
# RAW_NAME in code/prepare_real.py) and rerun; files already present are not downloaded again.
set -e
cd "$(dirname "$0")"
source ./_env.sh
mkdir -p data/model_ready

KEYS="$*"
[ -n "$KEYS" ] || KEYS="$(python3 -c 'import json; print(" ".join(json.load(open("config.json"))["models"]))')"
TASKSETS="$(python3 -c 'import json; print(" ".join(json.load(open("config.json"))["tasksets"]))')"

echo "=== environment"
"$PY" - <<'PYEOF'
import importlib
for m in ("numpy", "pandas", "torch", "transformers", "vllm", "huggingface_hub", "sklearn", "xlrd"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:16s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:16s} MISSING -- {e}")
PYEOF

for K in $KEYS; do
    ID="$(python3 -c "import json, sys; c = json.load(open('config.json'))['models']; print(c[sys.argv[1]]['id'])" "$K")" \
        || { echo "unknown model key $K"; exit 1; }
    echo; echo "=== model $K: $ID"
    if "$PY" - "$ID" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
assert tok.chat_template, "the tokenizer has no chat template"
print("  ->", p)
PYEOF
    then
        date > "data/model_ready/$K"
    else
        rm -f "data/model_ready/$K"
        echo "  !! $K NOT downloaded; its jobs will not be submitted"
    fi
done

echo; echo "=== real tables (UCI / scikit-learn)"
# The credit-default file is an old .xls, which pandas reads through xlrd. If xlrd is missing it is
# installed into a project folder (data/_pylib), leaving the main environment untouched.
if ! "$PY" -c "import xlrd" 2>/dev/null; then
    echo "  installing xlrd into data/_pylib (only used by prepare_real.py)"
    "$PY" -m pip install --quiet --target data/_pylib "xlrd>=2.0.1"
fi
PYTHONPATH="$PWD/data/_pylib${PYTHONPATH:+:$PYTHONPATH}" "$PY" code/prepare_real.py

for TS in $TASKSETS; do
    echo; echo "=== task set $TS"
    "$PY" code/make_tasks.py --taskset "$TS"
done

for TS in $TASKSETS; do
    echo; echo "=== self-test on $TS (CPU, scripted agent)"
    "$PY" code/selftest.py --taskset "$TS" --n-tasks 6
done

echo; echo "ready models: $(ls data/model_ready | tr '\n' ' ')"
echo "Now: bash submit-all.sh"
