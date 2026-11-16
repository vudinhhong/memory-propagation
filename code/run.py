"""One job: one model on one task set, every requested memory condition in turn (model loaded once).

    python code/run.py --taskset v1 --model q07 --out results_v1_p2/q07                 # all four conditions
    python code/run.py --taskset v1 --model q07 --memconds notes fullctx --out ...      # only these

Files, per condition (names carry task set, agent version, model and condition):
    clean_<run>.jsonl.gz    one row per task, full records (forks are rebuilt from it)
    forks_<run>.jsonl.gz    one row per fork (null / sham / inject at each k)
    summary_<run>.json      counts, skipped injection points, timing, versions -- written last
with <run> = <taskset>_<agent>_<model>_<memcond>. A condition whose three files exist is skipped;
one whose clean file exists resumes at the forks. A condition that fails is logged and the job goes
on with the next one, then exits non-zero. --n-tasks and --backend mock add _first<n> / _mock to
the names, so a smoke run can never overwrite a real one.
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memprop import runner as R                                                    # noqa: E402
from memprop.common import (ROOT, job_name, load_config, log, read_jsonl_gz, run_name,  # noqa: E402
                            write_json, write_jsonl_gz)
from memprop.executor import Executor                                              # noqa: E402
from memprop.tasks import csv_links, load_tasks                                    # noqa: E402


def run_condition(cfg, args, memcond, backend, ex, tasks, out, suffix, vllm_version):
    name = run_name(args.taskset, cfg["agent_version"], args.model, memcond) + suffix
    clean_path = out / f"clean_{name}.jsonl.gz"
    forks_path = out / f"forks_{name}.jsonl.gz"
    summary_path = out / f"summary_{name}.json"
    if clean_path.exists() and forks_path.exists() and summary_path.exists():
        log(f"{name}: all outputs already present; skipped")
        return
    log(f"=== condition {memcond}: {name}")
    t_start = time.time()
    too_long_before = backend.n_too_long
    ctx = R.Ctx(cfg, args.taskset, memcond, backend, ex, tasks)
    if clean_path.exists():
        clean_rows = read_jsonl_gz(clean_path)
        if sorted(r["task"] for r in clean_rows) != sorted(t["id"] for t in tasks):
            raise RuntimeError(f"{clean_path} holds a different task list; move it away and rerun")
        log(f"resuming from {clean_path.resolve()} ({len(clean_rows)} clean episodes)")
        clean_eps = R.episodes_from_clean_rows(clean_rows)
    else:
        clean_eps = R.run_clean(ctx, tasks)
        clean_rows = [R.grade_clean(ctx, ep, args.model) for ep in clean_eps]
        write_jsonl_gz(clean_path, clean_rows)
    t_clean = time.time() - t_start

    skipped, fork_rows = [], []
    if memcond != "nomem":
        forks, skipped = R.make_forks(ctx, clean_eps)
        n_inj = sum(len(t["injections"]) for t in tasks)
        log(f"forks: {len(forks)} ({len(forks) // len(R.ARMS)} injection points of {n_inj}; "
            f"{len(skipped)} skipped because the clean answer at k was wrong)")
        R.run_forks(ctx, forks)
        clean_by_task = {ep["task"]: ep for ep in clean_eps}
        fork_rows = [R.grade_fork(ctx, f, clean_by_task[f["task"]], args.model) for f in forks]
    write_jsonl_gz(forks_path, fork_rows)

    mdl = cfg["models"][args.model]
    acc = [s["correct"] for r in clean_rows for s in r["steps"]]
    write_json(summary_path, {
        "name": name, "taskset": args.taskset, "agent_version": cfg["agent_version"], "model_key": args.model,
        "model_id": mdl["id"], "memcond": memcond, "backend": args.backend, "vllm": vllm_version,
        "n_tasks": len(tasks), "clean_step_accuracy": sum(acc) / max(len(acc), 1),
        "n_injection_points": sum(len(t["injections"]) for t in tasks),
        "n_forks": len(fork_rows), "skipped_injection_points": skipped,
        "prompts_too_long": backend.n_too_long - too_long_before,
        "seconds_clean": round(t_clean), "seconds_total": round(time.time() - t_start),
        "config": cfg})
    log(f"{name} done in {time.time() - t_start:.0f}s")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskset", required=True, choices=sorted(cfg["tasksets"]))
    ap.add_argument("--model", required=True, choices=sorted(cfg["models"]))
    ap.add_argument("--memconds", nargs="+", default=cfg["memconds"], choices=cfg["memconds"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-tasks", type=int, default=0, help="first n tasks only (smoke runs)")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "mock"])
    args = ap.parse_args()

    tasks = load_tasks(ROOT, args.taskset)
    if args.n_tasks:
        tasks = tasks[:args.n_tasks]
    suffix = (f"_first{args.n_tasks}" if args.n_tasks else "") + ("_mock" if args.backend == "mock" else "")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jname = job_name(args.taskset, cfg["agent_version"], args.model) + suffix
    pending = [c for c in args.memconds
               if not all((out / f"{p}_{run_name(args.taskset, cfg['agent_version'], args.model, c)}{suffix}{e}").exists()
                          for p, e in (("clean", ".jsonl.gz"), ("forks", ".jsonl.gz"), ("summary", ".json")))]
    if not pending:
        log(f"{jname}: every requested condition is already complete in {out.resolve()}")
        return
    mdl = cfg["models"][args.model]
    log(f"job {jname}: model {mdl['id']}, conditions {' '.join(pending)}, {len(tasks)} tasks")

    workdir = Path(os.environ.get("SCRATCHDIR") or out) / f"exec_{jname}"
    ex = Executor(cfg["exec_workers"], cfg["exec_timeout_s"], cfg["exec_mem_gb"], workdir,
                  links=csv_links(tasks))                                          # before vLLM
    failed = []
    t0 = time.time()
    try:
        if args.backend == "vllm":
            from memprop.llm import VLLMChat
            sz = cfg["sizes"][mdl["size"]]
            backend = VLLMChat(mdl["id"], cfg["max_model_len"], cfg["llm_seed"], sz["gpu_memory_utilization"],
                               sz["tensor_parallel_size"])
        else:
            from memprop.llm import MockChat
            backend = MockChat(tasks)
        try:
            import vllm
            vllm_version = vllm.__version__
        except Exception:                                                          # noqa: BLE001
            vllm_version = None
        for memcond in pending:
            try:
                run_condition(cfg, args, memcond, backend, ex, tasks, out, suffix, vllm_version)
            except Exception:                                                      # noqa: BLE001
                failed.append(memcond)
                log(f"!!! condition {memcond} FAILED:\n{traceback.format_exc()}")
        backend.close()
    finally:
        ex.close()
    log(f"job {jname} finished in {time.time() - t0:.0f}s"
        + (f"; FAILED conditions: {' '.join(failed)}" if failed else ""))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
