# Error propagation through memory in data-analysis agents

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24274470.svg)](https://doi.org/10.5281/zenodo.24274470)

Code, tasks and run files for the paper *Counterfactual measurement of error
propagation through memory in language-model data-analysis agents*.

The experiment asks how often a single wrong value in a data-analysis agent's
record of an earlier step reaches its later results, and whether the form of
the record matters. Each task is a table plus eight dependent analysis steps
whose true values are known exactly. For any wrong input, the value every
later step would then take (the counterfactual) is known exactly as well. An
agent runs a task once cleanly. At every correct step `k` the episode is then
forked three ways, and steps `k+1..8` are regenerated:

* **null**: the record is untouched;
* **sham**: the record is rewritten with the agent's own value;
* **inject**: the record is rewritten with a plausible wrong value.

There are four memory conditions:

* `nomem`: no memory;
* `notes`: the agent's own one-line notes;
* `rawlog`: the printed output of each script;
* `fullctx`: each script and its output.

There are six models:

* `q01`, `q03`, `q07`, `q14`, `q32`: Qwen2.5-1.5B, 3B, 7B, 14B and
  32B-Instruct;
* `l08`: Llama-3.1-8B-Instruct.

Everything in the paper can be recomputed from the run files here without a
GPU.

## What is where

    config.json              task sets, models, tolerances, bootstrap size
    code/memprop/            the library
      tasks.py               table and chain generation, reference evaluator, counterfactuals
      agent.py               prompts, the four memory conditions, reply parsing
      executor.py            sandboxed execution of the agent's scripts
      llm.py                 vLLM backend and the scripted agent used by the self-test
      runner.py              clean runs, forks, grading
      analysis.py            task-level metrics, bootstrap (two-level on the real tables)
    code/make_tasks.py       builds a task set and re-verifies every stored value
    code/prepare_real.py     downloads and normalises the six public tables
    code/run.py              one model on one task set, all memory conditions
    code/selftest.py         CPU self-test with a scripted agent
    analyse.py               the tables of the paper (section 1-4 of each report)
    diagnose.py              why clean-run steps fail; --trace splits wrong values by cause
    make_figures.py          every figure of the paper, with figdata_*.json
    make_s1_table.py         S1 Table, from the four analyse.py reports in reports/
    make_s1_text.py          S1 Text: every prompt, rendered by the agent code for one task
    check-status.py          status of the cluster jobs; --resubmit for failed ones
    data/                    task definitions, synthetic tables, metadata of the public tables
    results_<taskset>_p2/    run files
    reports/                 the analysis and diagnostic outputs the paper quotes
    paper/                   figures, S1/S2 Table sources, paper/SI/ (supporting files as submitted)
    tools/                   packing the data on the cluster, writing and checking the manifest
    download.sh, submit-all.sh, job.sh, _env.sh, _job_body.sh, [R|V|VR][S|M|L].sh
                             cluster workflow (PBS); paths are site-specific, see _env.sh
    requirements.txt         packages for recomputing the results without a GPU
    requirements-cluster.txt pip freeze of the environment that produced the runs, for reference
                             (two entries are local builds and cannot be installed from it)
    MANIFEST.sha256          SHA-256 of every released file (python3 tools/make_manifest.py --check)

Task sets:

| name     | tables                          | tasks | injection points | role |
|----------|---------------------------------|-------|------------------|------|
| `v2`     | 240 synthetic tables            | 240   | 803              | main results |
| `v2real` | six public tables, 40 chains each | 240 | 927              | robustness |
| `v1`, `v1real` | same chains, original wording of the OLS step | 240 + 240 | 803 + 927 | wording sensitivity (S1 Table, S2 Fig) |

`v2` differs from `v1` only in the text of the OLS-predict steps, and
`make_tasks.py` verifies this. The original wording, "fit a line y = a + b * x",
led models to swap the two coefficients returned by `np.polyfit`. The
analysis plan for `v2` was fixed before it ran.

## Recomputing the results without a GPU

Python 3.10 with numpy, pandas and matplotlib (`pip install -r requirements.txt`).
Rebuilding the public tables with `code/prepare_real.py` also needs
scikit-learn (diabetes data) and xlrd (the credit-default .xls file).

    python analyse.py --taskset v2          # Table 3, all fork measures and controls
    python analyse.py --taskset v2real
    python make_figures.py                  # Figs 1-10, S1/S2 Fig (pdf, png, tif) -> paper/figures/; S2 Table -> paper/
    python make_s1_table.py                 # S1 Table -> paper/S1_Table.tex
    python make_s1_text.py                  # S1 Text (prompts) -> paper/S1_Text.tex
    python diagnose.py --taskset v2 --models q14 q32 --memconds notes rawlog fullctx --trace

`analyse.py` never overwrites an existing report: a new report gets a time
stamp. `diagnose.py --trace` on the real task sets needs the public tables,
which are not redistributed here. `python code/prepare_real.py` downloads them
and stops if a raw file differs from the SHA-256 recorded in
`data/real/<name>.json` (the diabetes data are read through scikit-learn and
have no recorded raw-file checksum).

### Run files

`results_<taskset>_p2/<model>/` holds, for every memory condition:

* `clean_<run>.jsonl.gz`: one row per task. It holds every step's answer and
  grade, the agent's scripts, outputs and notes, and the final report.
* `forks_<run>.jsonl.gz`: one row per fork (null, sham and inject at each
  injection point), with the regenerated steps graded against the true and
  counterfactual values.
* `summary_<run>.json`: counts, skipped injection points, timing and package
  versions.

`<run>` is `<taskset>_p2_<model>_<memcond>`. `p2` is the agent version used
for every result in the paper.

## Rerunning from scratch

    bash download.sh                         # models, public tables, task sets, CPU self-test
    TASKSETS="v2 v2real" bash submit-all.sh  # one PBS job per (task set, model)
    python3 check-status.py                  # which runs are done, running or failed

The released run files sit in `results_*_p2/`. Existing outputs are never
overwritten, so move them aside (or change `results_dir` in `config.json`)
before rerunning. The self-test and the runs need a POSIX system (Linux or
macOS) and, for the runs, vLLM and a GPU. The job scripts assume a PBS cluster;
set `USER_STORAGE` (model cache, Python environment) before running them
elsewhere. Without PBS, call `code/run.py` directly:

    python code/run.py --taskset v2 --model q07 --memconds nomem notes rawlog fullctx \
        --out results_v2_p2/q07

Decoding is greedy, but batched inference is not bit-exact. The large models
reproduce a clean run in at least 97.8% of regenerated steps. The smallest
model reproduces it in 73-87%, depending on the condition. This is the
"null drift" the paper reports.

## Not included

* The six public tables (Iris, Wine, diabetes, Cleveland heart disease,
  credit-card default, steel-plate faults). `code/prepare_real.py` fetches
  them from their public sources.
* The runs of the first agent version (p1). Its prompt let small models write
  the answer line as code that prints nothing, so those runs are invalid.
* Job logs and model weights.

## Citation

Archive: https://doi.org/10.5281/zenodo.24274470 (all versions; see `CITATION.cff`).

## Licence

Code under the MIT License. Tasks, synthetic tables and run files under
CC BY 4.0. See `LICENSE`.
