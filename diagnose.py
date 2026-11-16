#!/usr/bin/env python3
"""Why do steps fail in a clean run? Reads clean_<run>.jsonl.gz only (no GPU, standard library).

    python3 diagnose.py --taskset v1 --models q01 q03 q07 --memconds nomem notes
    python3 diagnose.py --taskset v1 --models q03 --memconds nomem --show 3     # print 3 failing scripts per cause
    python diagnose.py --taskset v1 --models q32 --memconds notes rawlog --trace  # needs numpy and pandas

Every step of a clean run is one of:
  correct          answer printed and equal to the true value
  wrong value      answer printed, but not the true value
  no code block    the reply had no ```python block (and no bare script)
  no ANSWER line   the script ran but printed no `ANSWER: <number>`
  timeout          the script exceeded the time limit
  prompt too long  the prompt did not fit the context window
  <ExceptionType>  the script raised (KeyError, SyntaxError, ...)

--trace (memory conditions only) splits the "wrong value" steps that use earlier results. The step is
recomputed from the data with the earlier results replaced by the values the agent RECORDED for them
(its note in `notes`, its printed answer in `rawlog` / `fullctx`):
  from wrong upstream     the answer equals that recomputation and a recorded upstream value is wrong:
                          the error came in through memory (natural propagation in the clean run)
  from rounded upstream   the answer equals that recomputation and every recorded upstream value is
                          correct within tolerance: the agent computed correctly from a rounded value,
                          and only the rounding moved the result (precision lost in memory)
  own error               the answer does not follow from the recorded values: a mistake in this step
    of which "OLS coefficients swapped": a `predict` step whose answer equals slope + intercept * x,
                          i.e. `a, b = np.polyfit(x, y, 1)` read as (intercept, slope) -- np.polyfit
                          returns the slope first, while the v1 step text writes the line as y = a + b * x
    of which "OLS without intercept": a `predict` step whose answer equals b0 * x with
                          b0 = sum(xy) / sum(xx), a line forced through the origin, e.g.
                          np.linalg.lstsq(X, y) where X has no column of ones
  upstream not recorded   an earlier step has no recorded value
"""
import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cause(step, rec):
    if step["correct"]:
        return "correct"
    if step["ok"]:
        return "wrong value"
    if rec.get("too_long"):
        return "prompt too long"
    err = rec.get("error") or ""
    if "no Python code block" in err:
        return "no code block"
    if "TimeoutError" in err:
        return "timeout"
    if not err.strip():
        return "no ANSWER line"
    m = re.findall(r"^(\w+(?:Error|Exception|Exit|Interrupt))\b", err, re.M)
    return m[-1] if m else "other error"


class Tracer:
    def __init__(self, taskset):
        sys.path.insert(0, str(ROOT / "code"))
        from memprop.common import close, load_config
        from memprop.tasks import compute_step, load_tasks, task_df
        self.close, self.compute_step, self.task_df = close, compute_step, task_df
        cfg = load_config()
        self.rtol, self.atol = cfg["rtol"], cfg["atol"]
        self.tasks = {t["id"]: t for t in load_tasks(ROOT, taskset)}
        self.dfs = {}

    def df(self, tid):
        if tid not in self.dfs:
            self.dfs[tid] = self.task_df(self.tasks[tid])
        return self.dfs[tid]

    def swapped(self, step, tid, vals, answer):
        import numpy as np
        d = self.df(tid)[[step["x"], step["y"]]].dropna()
        slope, intercept = np.polyfit(d[step["x"]].to_numpy(float), d[step["y"]].to_numpy(float), 1)
        x0 = vals[step["refs"][0] - 1]
        candidates = [x0, self.tasks[tid]["truth"][step["refs"][0] - 1]]
        return any(self.close(answer, slope + intercept * c, self.rtol, self.atol) for c in candidates)

    def no_intercept(self, step, tid, vals, answer):
        """Answer equals b0 * x0 with b0 = sum(xy) / sum(xx): a line forced through the origin
        (e.g. np.linalg.lstsq(X, y) with X holding only the x column, no column of ones)."""
        d = self.df(tid)[[step["x"], step["y"]]].dropna()
        x, y = d[step["x"]].to_numpy(float), d[step["y"]].to_numpy(float)
        sxx = float((x * x).sum())
        if sxx == 0:
            return False
        b0 = float((x * y).sum()) / sxx
        candidates = [vals[step["refs"][0] - 1], self.tasks[tid]["truth"][step["refs"][0] - 1]]
        return any(self.close(answer, b0 * c, self.rtol, self.atol) for c in candidates)

    def report(self, rows, memcond, show):
        cnt, ex = Counter(), {}
        n_wrong_ref = 0
        for r in rows:
            task = self.tasks[r["task"]]
            recorded = [(rec.get("note_value") if memcond == "notes" else rec.get("answer")) for rec in r["records"]]
            for s, rec in zip(r["steps"], r["records"]):
                step = task["steps"][s["j"] - 1]
                if s["correct"] or not s["ok"] or not step["refs"]:
                    continue
                n_wrong_ref += 1
                ups = {i: recorded[i - 1] for i in step["refs"]}
                if any(v is None for v in ups.values()):
                    k = "upstream not recorded"
                else:
                    vals = list(task["truth"])
                    for i, v in ups.items():
                        vals[i - 1] = v
                    try:
                        v = self.compute_step(step, self.df(r["task"]), vals)
                    except Exception:                               # noqa: BLE001
                        v = None
                    if v is not None and self.close(s["answer"], v, self.rtol, self.atol):
                        ok_up = all(self.close(x, task["truth"][i - 1], self.rtol, self.atol) for i, x in ups.items())
                        k = "from rounded upstream" if ok_up else "from wrong upstream"
                    elif step["type"] == "predict" and self.swapped(step, r["task"], vals, s["answer"]):
                        k = "own error: OLS coefficients swapped"
                    elif step["type"] == "predict" and self.no_intercept(step, r["task"], vals, s["answer"]):
                        k = "own error: OLS without intercept"
                    else:
                        k = f"own error: {step['type']}"
                cnt[k] += 1
                ex.setdefault(k, []).append((r["task"], s["j"], {i: (x, task["truth"][i - 1]) for i, x in ups.items()},
                                             s["answer"], task["truth"][s["j"] - 1]))
        print(f"  --trace: {n_wrong_ref} wrong values at steps that use earlier results")
        for k, v in sorted(cnt.items(), key=lambda kv: (not kv[0].startswith("from"), -kv[1])):
            print(f"    {k:36s} {v:5d}  {100 * v / max(n_wrong_ref, 1):5.1f}%")
        for k, items in ex.items():
            for task_id, j, ups, ans, tru in items[:show]:
                up = ", ".join(f"step {i}: recorded {x!r} true {t:.6g}" for i, (x, t) in ups.items())
                print(f"      {k}: {task_id} step {j}: answer {ans!r} true {tru:.6g}; {up}")


def main():
    cfg = json.loads((ROOT / "config.json").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskset", required=True, choices=list(cfg["tasksets"]))
    ap.add_argument("--models", nargs="+", default=list(cfg["models"]))
    ap.add_argument("--memconds", nargs="+", default=cfg["memconds"])
    ap.add_argument("--show", type=int, default=0, help="print this many failing scripts per cause")
    ap.add_argument("--trace", action="store_true", help="split wrong values by where they came from (needs pandas)")
    args = ap.parse_args()
    tracer = Tracer(args.taskset) if args.trace else None
    res = ROOT / cfg["tasksets"][args.taskset]["results_dir"]
    for m in args.models:
        for c in args.memconds:
            run = f"{args.taskset}_{cfg['agent_version']}_{m}_{c}"
            p = res / m / f"clean_{run}.jsonl.gz"
            if not p.exists():
                print(f"{run}: no clean file")
                continue
            with gzip.open(p, "rt") as f:
                rows = [json.loads(l) for l in f if l.strip()]
            counts, by_step, retried, examples = Counter(), Counter(), 0, {}
            n = 0
            for r in rows:
                for s, rec in zip(r["steps"], r["records"]):
                    k = cause(s, rec)
                    counts[k] += 1
                    n += 1
                    if k != "correct":
                        by_step[s["j"]] += 1
                    retried += rec.get("attempts", 1) > 1
                    examples.setdefault(k, []).append((r["task"], s["j"], rec))
            print(f"\n=== {run}   {n} steps, {retried} needed a retry")
            for k, v in counts.most_common():
                print(f"  {k:18s} {v:5d}  {100 * v / n:5.1f}%")
            print("  not correct by step: " + "  ".join(f"{j}:{by_step[j]}" for j in sorted(by_step)))
            if tracer is not None and c != "nomem":
                tracer.report(rows, c, args.show)
            if args.show:
                for k, ex in examples.items():
                    if k == "correct":
                        continue
                    for task, j, rec in ex[:args.show]:
                        print(f"\n  --- {k}: {task} step {j}")
                        print("  code:   " + (rec.get("code") or "(none)").strip().replace("\n", "\n          ")[:1500])
                        print("  stdout: " + (rec.get("stdout") or "").strip().replace("\n", "\n          ")[:400])
                        print("  error:  " + (rec.get("error") or "").strip().replace("\n", "\n          ")[-600:])


if __name__ == "__main__":
    main()
