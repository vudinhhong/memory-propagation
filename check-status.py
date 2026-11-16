#!/usr/bin/env python3
"""Status of every job (one per task set x model) and of each memory condition inside it.

    python3 check-status.py                 # every task set
    python3 check-status.py --taskset v1real
    python3 check-status.py -v              # also print the last log lines of failed jobs
    python3 check-status.py --resubmit      # resubmit jobs with unfinished conditions (asks first)

Run it on the login node; it needs only the standard library. For each condition:
  DONE       summary_<run>.json exists (written last, after clean_ and forks_)
  QUEUED / RUNNING   the model's job is still in PBS (its output log points into the model's directory)
  FAILED     the job is no longer in PBS and this condition has no summary; the reason is read from the
             logs (selftest, Python traceback, walltime, memory, CUDA OOM, a condition marked FAILED...)
  PARTIAL    like FAILED, but its clean phase finished: a resubmitted job resumes from clean_<run>
  NOT SUBMITTED   no directory for the model
When qstat is not available, a job whose realtime log changed in the last 15 minutes is "RUNNING?".

--resubmit moves the PBS logs and the realtime log of each such job into <dir>/attempt_<time>/ (so the
failure stays on disk), keeps every finished file, and calls job.sh with only the unfinished conditions.
"""
import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PATTERNS = [   # (reason, regex) -- first match wins, checked on error log + realtime log + output log
    ("walltime exceeded", r"walltime \d+ exceeded limit|PBS: job killed: walltime"),
    ("host memory exceeded", r"mem \d+\w* exceeded limit|PBS: job killed: mem|Out of memory|oom-kill"),
    ("CUDA out of memory", r"CUDA out of memory|torch\.OutOfMemoryError|CUDA error: out of memory"),
    ("not enough GPU memory for vLLM", r"No available memory for the cache blocks|not enough KV cache|"
                                        r"larger than the maximum number of tokens that can be stored in KV cache"),
    ("selftest failed", r"SELFTEST FAILED|ERROR: selftest failed"),
    ("bad parameters / missing input", r"bad PARAMS|unknown model key|unknown memory condition|unknown task set|"
                                       r"is submitted with|missing data/tasks_|was not downloaded"),
    ("model could not be loaded", r"OSError: .*(does not appear to have|not a valid model|gated repo)|"
                                  r"401 Client Error|Cannot access gated repo"),
    ("a condition raised (see '!!! condition ... FAILED' in the realtime log)", r"!!! condition \w+ FAILED"),
    ("python error", r"Traceback \(most recent call last\)"),
    ("run failed", r"ERROR: run failed"),
]


def pbs_jobs():
    """{output-log directory (resolved): (job id, state)} from qstat -f, or None if PBS is not reachable."""
    user = os.environ.get("USER", "")
    try:
        ids = subprocess.run(["qselect", "-u", user], capture_output=True, text=True, timeout=60).stdout.split()
    except Exception:                                               # noqa: BLE001
        return None
    if not ids:
        return {}
    try:
        out = subprocess.run(["qstat", "-f"] + ids, capture_output=True, text=True, timeout=120).stdout
    except Exception:                                               # noqa: BLE001
        return None
    jobs, cur = {}, {}
    for line in re.sub(r"\n\t", "", out).splitlines():              # qstat -f wraps long values with newline+tab
        if line.startswith("Job Id:"):
            cur = {"id": line.split(":", 1)[1].strip()}
        elif "=" in line and cur:
            k, v = [x.strip() for x in line.split("=", 1)]
            cur[k] = v
            if "Output_Path" in cur and "job_state" in cur and k in ("Output_Path", "job_state"):
                path = cur["Output_Path"].split(":", 1)[-1]
                jobs[str(Path(path).parent.resolve())] = (cur["id"], cur["job_state"])
    return jobs


def logs(d, job):
    return sorted(d.glob("*_error.log")) + [d / f"realtime_{job}.log"] + sorted(d.glob("*_output.log"))


def reason(d, job, memcond):
    blob = "\n".join(p.read_text(errors="replace") for p in logs(d, job) if p.exists())
    if re.search(rf"!!! condition {memcond} FAILED", blob):
        m = re.search(rf"!!! condition {memcond} FAILED:\n(.*?)(?:\n\d\d:\d\d:\d\d |\Z)", blob, re.S)
        last = [l for l in (m.group(1).splitlines() if m else []) if l.strip()]
        return "condition raised: " + (last[-1].strip() if last else "see realtime log")
    for name, rx in PATTERNS:
        if re.search(rx, blob):
            return name
    if not blob.strip():
        return "no log at all (job never started, or PBS logs not copied back yet)"
    return "ended without a summary (no known error message; see logs)"


def done_info(path):
    s = json.loads(path.read_text())
    note = []
    if s.get("prompts_too_long"):
        note.append(f"{s['prompts_too_long']} prompts too long")
    if s.get("backend") != "vllm":
        note.append(f"backend {s.get('backend')}")
    return (f"clean acc {100 * s['clean_step_accuracy']:5.1f}%  forks {s['n_forks']:5d}  "
            f"skipped k {len(s['skipped_injection_points']):4d}  {s['seconds_total'] / 3600:5.2f} h"
            + (f"  [{'; '.join(note)}]" if note else ""))


def n_rows(path):
    try:
        with gzip.open(path, "rt") as f:
            return sum(1 for _ in f)
    except Exception:                                               # noqa: BLE001
        return None


def main():
    cfg = json.loads((ROOT / "config.json").read_text())
    agent = cfg["agent_version"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskset", nargs="+", default=list(cfg["tasksets"]), choices=list(cfg["tasksets"]))
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--resubmit", action="store_true")
    args = ap.parse_args()

    jobs = pbs_jobs()
    if jobs is None:
        print("(qselect/qstat not available: running jobs are guessed from log times)\n")
    counts, to_resubmit = {}, []
    for ts in args.taskset:
        tc = cfg["tasksets"][ts]
        res = ROOT / tc["results_dir"]
        print(f"=== task set {ts}, agent {agent}   ({res})")
        for m, mc in cfg["models"].items():
            script = f"{tc['job_prefix']}{mc['size']}.sh"
            job = f"{ts}_{agent}_{m}"
            d = res / m
            in_pbs = jobs.get(str(d.resolve())) if (jobs is not None and d.exists()) else None
            rt = d / f"realtime_{job}.log"
            guess_running = (jobs is None and rt.exists() and time.time() - rt.stat().st_mtime < 900)
            head = f"  {m}  ({script})"
            if not d.exists():
                head += "  NOT SUBMITTED"
            elif in_pbs:
                state = {"Q": "QUEUED", "R": "RUNNING", "H": "HELD", "E": "EXITING"}.get(in_pbs[1], f"PBS:{in_pbs[1]}")
                head += f"  job {state} {in_pbs[0]}"
            elif guess_running:
                head += f"  job RUNNING? (log updated {int(time.time() - rt.stat().st_mtime)}s ago)"
            print(head)
            unfinished = []
            for c in cfg["memconds"]:
                run = f"{ts}_{agent}_{m}_{c}"
                summ, cp = d / f"summary_{run}.json", d / f"clean_{run}.jsonl.gz"
                if not d.exists():
                    st, info = "NOT SUBMITTED", ""
                elif summ.exists():
                    st, info = "DONE", done_info(summ)
                elif in_pbs or guess_running:
                    st = "QUEUED" if in_pbs and in_pbs[1] == "Q" else "RUNNING"
                    info = f"clean phase done ({n_rows(cp)} tasks), forks running" if cp.exists() else ""
                else:
                    st = "PARTIAL" if cp.exists() else "FAILED"
                    info = reason(d, job, c)
                    if cp.exists():
                        info += f"  (clean phase done: {n_rows(cp)} tasks; a rerun resumes from it)"
                    unfinished.append(c)
                counts[st] = counts.get(st, 0) + 1
                print(f"      {c:8s} {st:14s} {info}")
            if unfinished:
                to_resubmit.append((ts, m, d, script, job, unfinished))
                if args.verbose:
                    for p in logs(d, job):
                        try:
                            lines = p.read_text(errors="replace").splitlines()[-12:]
                        except Exception:                           # noqa: BLE001
                            lines = []
                        if lines:
                            print(f"      --- {p.name}")
                            for l in lines:
                                print(f"      | {l}")
        print()
    print("conditions: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    if to_resubmit:
        print("\nto resubmit by hand (PBS overwrites old logs -- --resubmit archives them first):")
        for ts, m, d, script, job, unf in to_resubmit:
            print(f"  ./job.sh {script} {d.relative_to(ROOT)} {m} {' '.join(unf)}")
    if args.resubmit and to_resubmit:
        if jobs is None:
            sys.exit("\n--resubmit needs qstat, to be sure none of these is still running. Nothing submitted.")
        if input(f"\nresubmit {len(to_resubmit)} job(s)? [y/N] ").strip().lower() != "y":
            print("nothing submitted")
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for ts, m, d, script, job, unf in to_resubmit:
            arch = d / f"attempt_{stamp}"
            arch.mkdir()
            for p in logs(d, job):
                if p.exists():
                    shutil.move(str(p), str(arch / p.name))
            print(f"  {job}: logs moved to {arch.relative_to(ROOT)}")
            subprocess.run(["./job.sh", script, str(d.relative_to(ROOT)), m] + unf, input="\n", text=True,
                           cwd=ROOT, check=False)
            time.sleep(2)


if __name__ == "__main__":
    main()
