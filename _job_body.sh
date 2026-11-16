# Shared body of the PBS scripts <prefix><size>.sh (S/M/L for v1, RS/RM/RL for v1real, VS/VM/VL for v2,
# VRS/VRM/VRL for v2real); sourced. Those scripts differ only in the #PBS resource lines, SIZE (the
# model's "size" key in config.json) and TASKSET.
#
# PARAMS (passed by job.sh) = "<model> [condition ...]", e.g. "q07" or "q07 notes fullctx":
#   model       key of config.json "models" (q01 q03 q07 l08 q14 q32)
#   condition   nomem | notes | rawlog | fullctx; empty = all four
# One job = one task set x one model: the model is loaded once and the conditions run in turn, each on
# all 240 tasks (clean phase, then forks). Finished conditions are skipped and a condition whose clean
# phase is done resumes from it, so resubmitting the same command is safe.
# Every output file is named <taskset>_<agent>_<model>_<condition> (see code/run.py).

cd "$PBS_O_WORKDIR" || { echo "cannot cd to PBS_O_WORKDIR=$PBS_O_WORKDIR"; exit 2; }
source ./_env.sh
export CUDA_VISIBLE_DEVICES="0"

read -r -a P <<< "$PARAMS"
if [ "${#P[@]}" -lt 1 ]; then
    echo "bad PARAMS '$PARAMS' (expected '<model> [memcond ...]', e.g. 'q07' or 'q07 notes')"; exit 2
fi
MODEL="${P[0]}"
MEMS="${P[*]:1}"
# check the arguments against config.json before spending time loading the model
$PY - "$TASKSET" "$SIZE" "$MODEL" $MEMS <<'PYEOF' || exit 2
import json, sys
c = json.load(open("config.json"))
ts, size, m, *mems = sys.argv[1:]
if ts not in c["tasksets"]:
    sys.exit(f"unknown task set '{ts}' (config.json tasksets: {', '.join(c['tasksets'])})")
if m not in c["models"]:
    sys.exit(f"unknown model key '{m}' (config.json models: {', '.join(c['models'])})")
for mem in mems:
    if mem not in c["memconds"]:
        sys.exit(f"unknown memory condition '{mem}' (config.json memconds: {', '.join(c['memconds'])})")
if c["models"][m]["size"] != size:
    want = c["tasksets"][ts]["job_prefix"] + c["models"][m]["size"]
    sys.exit(f"model {m} on task set {ts} is submitted with {want}.sh, not this script")
PYEOF
[ -n "$MEMS" ] || MEMS="$($PY -c 'import json; print(" ".join(json.load(open("config.json"))["memconds"]))')"
AGENT="$($PY -c 'import json; print(json.load(open("config.json"))["agent_version"])')"
JOB="${TASKSET}_${AGENT}_${MODEL}"

test -s "data/tasks_${TASKSET}.json" || { echo "missing data/tasks_${TASKSET}.json -- run download.sh first"; exit 2; }
test -f "data/model_ready/${MODEL}" || { echo "model ${MODEL} was not downloaded -- run download.sh first"; exit 2; }

echo "=========================================================="
echo "JOB             : $JOB  (taskset=$TASKSET agent=$AGENT model=$MODEL size=$SIZE)"
echo "CONDITIONS      : $MEMS"
echo "OUTPUT_PATH     : $OUTPUT_PATH"
echo "Node            : $(hostname)"
echo "Scratch         : $SCRATCHDIR"
echo "Started         : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=========================================================="
mkdir -p "$OUTPUT_PATH"
LOG="$OUTPUT_PATH/realtime_${JOB}.log"

run() { PYTHONUNBUFFERED=1 "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

# CPU check of the whole pipeline with the scripted agent, on this job's conditions (a few minutes).
run $PY code/selftest.py --taskset "$TASKSET" --n-tasks 3 --memconds $MEMS --out "${SCRATCHDIR:-$OUTPUT_PATH}/selftest_${JOB}" \
    || { echo "ERROR: selftest failed"; exit 1; }

run $PY code/run.py --taskset "$TASKSET" --model "$MODEL" --memconds $MEMS --out "$OUTPUT_PATH" \
    || { echo "ERROR: run failed (see 'FAILED' in $LOG; finished conditions are kept)"; exit 1; }

echo "=========================================================="
echo "JOB $JOB DONE AT: $(date)"
echo "=========================================================="
