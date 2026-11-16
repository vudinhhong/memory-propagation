# Environment shared by the job scripts (sourced from _job_body.sh) and by download.sh.
# USER_STORAGE holds the model cache, the Python environment and the Hugging Face token. The runs used
# a cluster home directory; elsewhere set USER_STORAGE (and PY) before running, or a local .cache/ is used.
if [ -z "$USER_STORAGE" ]; then
    if [ -d /storage/brno2/home/vdhong ]; then USER_STORAGE="/storage/brno2/home/vdhong"; else USER_STORAGE="$PWD/.cache"; fi
fi
export HF_HOME="$USER_STORAGE/hf_models_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
unset HUGGINGFACE_HUB_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export XDG_CACHE_HOME="$USER_STORAGE/vllm_compile_cache"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME"
# Inside a PBS job, HOME points to a writable folder of USER_STORAGE.
if [ -n "$PBS_O_WORKDIR" ]; then export HOME="$USER_STORAGE/my_fake_home"; fi
# Inside a job TMPDIR is the scratch directory. The test is needed because download.sh also sources this
# file on the login node, where SCRATCHDIR is empty and TMPDIR="" would break pip and tempfile.
if [ -n "$SCRATCHDIR" ]; then export TMPDIR="$SCRATCHDIR" TEMP="$SCRATCHDIR" TMP="$SCRATCHDIR"; fi
# CUDA_VISIBLE_DEVICES="0" is set in _job_body.sh (only jobs have a GPU).
export CUDA_HOME=/usr/local/cuda
export PATH="$USER_STORAGE/.conda/envs/vllm_env/bin:$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Several jobs can share a node (ncpus=8). Without a limit every process opens hundreds of threads and
# later jobs fail for lack of threads. OPENBLAS is included because pip's numpy uses OpenBLAS. When the
# pool that runs agent code starts, executor.py lowers OMP/MKL/OPENBLAS to 1 for the whole process
# (6 workers x 1 thread on 8 CPUs); vLLM sets one torch thread per worker itself.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 RAYON_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
# The main process has already started a process pool for agent code; vLLM starts its own processes
# with spawn so that it never forks a multi-threaded process.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
if [ -f "$USER_STORAGE/.hf_token" ]; then export HF_TOKEN="$(cat "$USER_STORAGE/.hf_token")"; fi
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME"
if [ -n "$PBS_O_WORKDIR" ]; then mkdir -p "$HOME"; fi
if [ -z "$PY" ]; then
    PY="$USER_STORAGE/.conda/envs/vllm_env/bin/python"
    [ -x "$PY" ] || PY="$(command -v python3)"
fi
