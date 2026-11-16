#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=40gb:mem=48gb:scratch_local=20gb
#PBS -l walltime=12:00:00
#
# Public-table task set v2real (v1real with the revised OLS wording). Models up to 8B (q01 q03 q07 l08). At most 16 GB of bf16 weights; the rest of the GPU holds the KV cache.
# Arguments passed through job.sh: "<model> [condition ...]" (empty = all four conditions), e.g.
#   ./job.sh VRS.sh results_v2real_p2/<model> <model>
# Normally not called by hand: submit-all.sh picks the script from the task set and the model size.
SIZE=S
TASKSET=v2real
# PBS starts the script in the home directory, so cd before sourcing _job_body.sh.
cd "$PBS_O_WORKDIR" || { echo "cannot cd to PBS_O_WORKDIR=$PBS_O_WORKDIR"; exit 2; }
source ./_job_body.sh
