#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:gpu_mem=80gb:mem=96gb:scratch_local=20gb
#PBS -l walltime=48:00:00
#
# Public-table task set v1real (six public tables). 32B model (q32). 65 GB of weights, fits only on an 80 GB card (gpu_memory_utilization 0.92); the slowest job.
# Arguments passed through job.sh: "<model> [condition ...]" (empty = all four conditions), e.g.
#   ./job.sh RL.sh results_v1real_p2/<model> <model>
# Normally not called by hand: submit-all.sh picks the script from the task set and the model size.
SIZE=L
TASKSET=v1real
# PBS starts the script in the home directory, so cd before sourcing _job_body.sh.
cd "$PBS_O_WORKDIR" || { echo "cannot cd to PBS_O_WORKDIR=$PBS_O_WORKDIR"; exit 2; }
source ./_job_body.sh
