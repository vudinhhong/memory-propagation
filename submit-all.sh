#!/bin/bash
# Submit the jobs: one job = one task set x one model, running the four memory conditions in turn
# (the model is loaded once). Four task sets x six models = 24 jobs; the paper's main runs are v2 and v2real.
#
#   bash download.sh                           # once, beforehand
#   bash submit-all.sh                         # every task set, every downloaded model
#   bash submit-all.sh q07 q14                 # selected models
#   TASKSETS="v2 v2real" bash submit-all.sh    # selected task sets
#
# PBS script = <job_prefix of the task set><model size>.sh: S/M/L.sh for v1, RS/RM/RL.sh for v1real,
# VS/VM/VL.sh for v2, VRS/VRM/VRL.sh for v2real.
# The jobs are independent (they read only what download.sh created and write only to their own
# directory), so they are submitted together. An existing results directory means the job was already
# submitted; it is not submitted again, so two jobs never write to the same place. For a failed job:
# python3 check-status.py --resubmit (keeps the old logs, runs only the missing conditions).
set -e
cd "$(dirname "$0")"
cfg() { python3 -c "import json; c = json.load(open('config.json')); $1"; }
MODELS="$*"
[ -n "$MODELS" ] || MODELS="$(cfg 'print(" ".join(c["models"]))')"
[ -n "$TASKSETS" ] || TASKSETS="$(cfg 'print(" ".join(c["tasksets"]))')"

for TS in $TASKSETS; do
    cfg "import sys; sys.exit(0 if '$TS' in c['tasksets'] else 1)" || { echo "unknown task set '$TS'"; exit 1; }
    test -s "data/tasks_${TS}.json" || { echo "STOP: data/tasks_${TS}.json missing -- run 'bash download.sh' first. Nothing submitted."; exit 1; }
done
for M in $MODELS; do
    cfg "import sys; sys.exit(0 if '$M' in c['models'] else 1)" || { echo "unknown model key '$M'"; exit 1; }
done

n=0
for TS in $TASKSETS; do
    RESULTS="$(cfg "print(c['tasksets']['$TS']['results_dir'])")"
    PREFIX="$(cfg "print(c['tasksets']['$TS']['job_prefix'])")"
    mkdir -p "$RESULTS"
    for M in $MODELS; do
        if [ ! -f "data/model_ready/$M" ]; then echo "skip $M: not downloaded (bash download.sh $M)"; continue; fi
        SCRIPT="${PREFIX}$(cfg "print(c['models']['$M']['size'])").sh"
        [ -f "$SCRIPT" ] || { echo "missing job script $SCRIPT"; exit 1; }
        out="$RESULTS/$M"
        if [ -e "$out" ]; then echo "skip $TS $M: $out already exists"; continue; fi
        printf '\n' | ./job.sh "$SCRIPT" "$out" "$M"
        n=$((n + 1))
        sleep 2
    done
done
echo
echo "Submitted $n job(s).   python3 check-status.py"
echo "When they finish:      for ts in $TASKSETS; do python analyse.py --taskset \$ts; done"
