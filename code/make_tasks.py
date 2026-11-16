"""Generate a task set once (CPU). download.sh calls it on the login node.

    python code/make_tasks.py --taskset v1         # synthetic tables, one per task
    python code/make_tasks.py --taskset v1real     # the six real tables (run prepare_real.py first)
    add --force to regenerate (only before any job has run on that task set)

Writes data/tasks_<taskset>.json (and, for synthetic sets, data/tasks_<taskset>/tNNN.csv), then
reloads every CSV from disk and recomputes all true values and counterfactual chains to check the
stored ones.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memprop.common import ROOT, load_config, log                                  # noqa: E402
from memprop.tasks import build_task, build_task_real, check_task, load_tasks, real_meta  # noqa: E402


def synthetic(cfg, ts, tcfg):
    csv_dir = ROOT / "data" / f"tasks_{ts}"
    csv_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i in range(tcfg["n_tasks"]):
        tid = f"t{i:03d}"
        task, csv_text = build_task(tid, tcfg["task_seed"] * 1000 + i, cfg["n_steps"], cfg["rtol"], cfg["atol"],
                                    cfg["moved_factor"], tcfg.get("predict_wording", "ab"))
        task["csv"] = f"data/tasks_{ts}/{tid}.csv"
        (ROOT / task["csv"]).write_text(csv_text)
        tasks.append(task)
        if (i + 1) % 40 == 0:
            log(f"  {i + 1}/{tcfg['n_tasks']} tasks")
    return tasks


def real(cfg, ts, tcfg):
    tasks = []
    for di, name in enumerate(tcfg["datasets"]):
        csv_rel = f"data/real/{name}.csv"
        info_path = ROOT / "data" / "real" / f"{name}.json"
        if not (ROOT / csv_rel).exists() or not info_path.exists():
            raise SystemExit(f"{csv_rel} is missing -- run 'python code/prepare_real.py' first")
        info = json.loads(info_path.read_text())
        df = pd.read_csv(ROOT / csv_rel)
        meta = real_meta(df, info)
        seen = set()
        for i in range(tcfg["per_dataset"]):
            task, sig = build_task_real(f"{name}_{i:02d}", tcfg["task_seed"] * 1000 + di * 100 + i, df, meta, name,
                                        cfg["n_steps"], cfg["rtol"], cfg["atol"], cfg["moved_factor"], seen,
                                        tcfg.get("predict_wording", "ab"))
            seen.add(sig)
            task["csv"] = csv_rel
            tasks.append(task)
        log(f"  {name}: {tcfg['per_dataset']} distinct chains (group '{meta['group_col']}', "
            f"filter levels {meta['big_levels']})")
    return tasks


def compare_chains(ts, other):
    """A task set declared `same_chains_as` another must differ from it only in step wording."""
    path = ROOT / "data" / f"tasks_{other}.json"
    if not path.exists():
        print(f"same_chains_as {other}: data/tasks_{other}.json not present, comparison skipped")
        return
    a = {t["id"]: t for t in load_tasks(ROOT, ts)}
    b = {t["id"]: t for t in load_tasks(ROOT, other)}
    if sorted(a) != sorted(b):
        raise SystemExit(f"{ts} and {other} have different task ids")
    changed, problems = Counter(), []
    for tid, t in a.items():
        u = b[tid]
        strip = lambda st: [{k: v for k, v in x.items() if k != "text"} for x in st]   # noqa: E731
        if strip(t["steps"]) != strip(u["steps"]):
            problems.append(f"{tid}: steps differ")
        if t["truth"] != u["truth"] or t["injections"] != u["injections"]:
            problems.append(f"{tid}: true values or injection points differ")
        if Path(t["csv_abs"]).read_bytes() != Path(u["csv_abs"]).read_bytes():
            problems.append(f"{tid}: data files differ")
        for x, y in zip(t["steps"], u["steps"]):
            if x["text"] != y["text"]:
                changed[x["type"]] += 1
    if problems:
        print("\n".join(problems[:20]))
        raise SystemExit(f"{ts} is declared same_chains_as {other} but differs in {len(problems)} places")
    print(f"same_chains_as {other}: identical tables, steps, true values and injection points; "
          f"wording changed in {dict(changed)} steps")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskset", required=True, choices=sorted(cfg["tasksets"]))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ts = args.taskset
    tcfg = cfg["tasksets"][ts]
    out_json = ROOT / "data" / f"tasks_{ts}.json"
    if out_json.exists() and not args.force:
        log(f"{out_json} already exists; not regenerating (use --force before any job has used it)")
        return
    tasks = synthetic(cfg, ts, tcfg) if tcfg["kind"] == "synthetic" else real(cfg, ts, tcfg)
    gen = dict(tcfg, **{k: cfg[k] for k in ("n_steps", "rtol", "atol", "moved_factor")})
    tmp = out_json.with_name(out_json.name + ".part")
    tmp.write_text(json.dumps({"taskset": ts, "generator": gen, "tasks": tasks}))
    tmp.replace(out_json)
    log(f"wrote {out_json.resolve()} ({len(tasks)} tasks)")

    loaded = load_tasks(ROOT, ts)
    problems = [p for t in loaded for p in check_task(t, cfg["rtol"], cfg["atol"])]
    if problems:
        print("\n".join(problems[:50]))
        raise SystemExit(f"{len(problems)} problems in task set {ts}")
    types = Counter(s["type"] for t in loaded for s in t["steps"])
    n_inj = [len(t["injections"]) for t in loaded]
    n_moved = [len(i["moved"]) for t in loaded for i in t["injections"]]
    depth = Counter(max(t["depth"]) for t in loaded)
    print(f"checked: every stored value and counterfactual recomputed from the CSV files ({len(loaded)} tasks)")
    print(f"step types           : {dict(sorted(types.items()))}")
    print(f"max depth per task   : {dict(sorted(depth.items()))}")
    print(f"injection points/task: min {min(n_inj)}  mean {sum(n_inj) / len(n_inj):.2f}  max {max(n_inj)}  "
          f"total {sum(n_inj)}")
    print(f"moved descendants    : mean {sum(n_moved) / len(n_moved):.2f} per injection point")
    print(f"rows per table       : {min(t['n_rows'] for t in loaded)}-{max(t['n_rows'] for t in loaded)}")
    if tcfg["kind"] == "real":
        print(f"tasks per dataset    : {dict(Counter(t['dataset'] for t in loaded))}")
    if tcfg.get("same_chains_as"):
        compare_chains(ts, tcfg["same_chains_as"])


if __name__ == "__main__":
    main()
