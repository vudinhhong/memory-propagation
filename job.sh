#!/bin/bash
# Submit one PBS job: ./job.sh <job script> <output dir> <model> [memcond ...]
#   e.g. ./job.sh VS.sh results_v2_p2/q07 q07
# submit-all.sh and check-status.py --resubmit call it; the PBS logs go to the output dir.
if [ "$#" -lt 3 ]; then
    echo "usage: ./job.sh <job script> <output dir> <model> [memcond ...]"
    exit 1
fi
SH_FILE="$1"
OUTPUT_PATH="$2"
shift 2
PARAMS="$*"
if [ ! -f "$SH_FILE" ]; then
    echo "ERROR: job script '$SH_FILE' not found"
    exit 1
fi
SAFE="${PARAMS// /_}"
SAFE="${SAFE//-/}"
mkdir -p "$OUTPUT_PATH"
OUTPUT_PATH=$(realpath "$OUTPUT_PATH")
JOB_NAME="$(basename "${SH_FILE%.*}")_${SAFE}_$(basename "${OUTPUT_PATH%/}")"
SH_FILE=$(realpath "$SH_FILE")
echo "job script : $SH_FILE"
echo "output dir : $OUTPUT_PATH"
echo "parameters : $PARAMS"
read -r -p "Press Enter to submit, or Ctrl+C to cancel..."
if ! qsub -N "$JOB_NAME" -o "$OUTPUT_PATH/${JOB_NAME}_output.log" -e "$OUTPUT_PATH/${JOB_NAME}_error.log" \
        -v OUTPUT_PATH="$OUTPUT_PATH",PARAMS="$PARAMS" "$SH_FILE"; then
    echo "ERROR: could not submit the job"
    exit 1
fi
echo "Job submitted."
