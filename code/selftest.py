"""End-to-end check on CPU, no model: task set, executor, prompts, forks, grading and analysis.

    python code/selftest.py --taskset v1             # 6 tasks, all memory conditions (login node)
    python code/selftest.py --taskset v1real --n-tasks 3 --memconds notes --out $SCRATCHDIR/selftest   (every job)

The tasks are spread evenly over the task set, so on v1real they come from different tables.

The agent is MockChat: a perfect analyst that reuses any value it can read in its history. So the
expected numbers are known exactly -- clean runs all correct, sham and null forks all correct,
every moved descendant in an inject fork equal to its counterfactual value, the report repeating
the injected value -- and any deviation means the pipeline, not the model, is wrong. It also
forces one crashing script and one endless loop, which the retry has to recover.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memprop import runner as R                                        # noqa: E402
from memprop.analysis import analyse                                   # noqa: E402
from memprop.common import ROOT, close, load_config, log, parse_answer, write_jsonl_gz  # noqa: E402
from memprop.executor import Executor                                  # noqa: E402
from memprop.llm import MockChat, reference_code                       # noqa: E402
from memprop.tasks import check_task, csv_links, load_tasks                       # noqa: E402

FAILS = []


def expect(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def rate(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskset", required=True, choices=sorted(cfg["tasksets"]))
    ap.add_argument("--n-tasks", type=int, default=6)
    ap.add_argument("--memconds", nargs="+", default=cfg["memconds"], choices=cfg["memconds"])
    ap.add_argument("--out", default=str(ROOT / "selftest_out"))
    args = ap.parse_args()
    if args.n_tasks < 2:
        raise SystemExit("--n-tasks must be at least 2 (tasks 0 and 1 carry the forced failures)")
    cfg = dict(cfg, exec_timeout_s=2, exec_workers=2)
    rtol, atol = cfg["rtol"], cfg["atol"]
    every = load_tasks(ROOT, args.taskset)
    tasks = [every[i] for i in np.linspace(0, len(every) - 1, args.n_tasks).round().astype(int)]
    tasks = list({t["id"]: t for t in tasks}.values())
    out = Path(args.out)

    log("1. task set: stored values reproduce from the CSV files")
    probs = [p for t in tasks for p in check_task(t, rtol, atol)]
    expect(not probs, f"{len(tasks)} tasks consistent" + (f" -- {probs[:3]}" if probs else ""))

    ex = Executor(cfg["exec_workers"], cfg["exec_timeout_s"], cfg["exec_mem_gb"], out / "exec", links=csv_links(tasks))
    try:
        log("2. reference scripts run in the executor and give the true values")
        codes, truth = [], []
        for t in tasks:
            for s in t["steps"]:
                codes.append(reference_code(s, t["csv_name"], [t["truth"][i - 1] for i in s["refs"]]))
            truth.extend(t["truth"])
        res = ex.run_many(codes)
        bad = [(c, r) for c, r, v in zip(codes, res, truth) if not close(parse_answer(r["stdout"]), v, rtol, atol)]
        expect(not bad, f"{len(codes)} reference scripts correct" + (f" -- first failure: {bad[0][1]}" if bad else ""))
        r_loop = ex.run_many(["while True:\n    pass\n"])[0]
        expect(not r_loop["ok"] and "Timeout" in r_loop["error"], "an endless loop is stopped by the time limit")
        r_exit = ex.run_many(["import sys\nprint('x')\nsys.exit(3)\n"])[0]
        expect(not r_exit["ok"] and r_exit["stdout"].startswith("x"), "sys.exit inside a script is contained")
        expect(ex.run_many(["print('ANSWER: 1')"])[0]["ok"], "the pool still works after both")

        clean_all, forks_all = [], []
        for mc in args.memconds:
            log(f"3. memory condition '{mc}': clean episodes, forks, grading")
            ctx = R.Ctx(cfg, args.taskset, mc, MockChat(tasks), ex, tasks)
            eps = R.run_clean(ctx, tasks)
            rows = [R.grade_clean(ctx, ep, "mock") for ep in eps]
            clean_all += rows
            acc = rate([s["correct"] for r in rows for s in r["steps"]])
            expect(acc is not None and acc >= 0.97, f"[{mc}] clean step accuracy {f3(acc)}")
            s03 = rows[0]["steps"][2]
            s12 = rows[1]["steps"][1]
            expect(s03["attempts"] == 2 and s03["correct"], f"[{mc}] crashing first attempt recovered on retry")
            expect(s12["attempts"] == 2 and s12["correct"], f"[{mc}] endless-loop first attempt recovered on retry")
            if mc == "notes":
                expect((rate([s["note_ok"] for r in rows for s in r["steps"]]) or 0) >= 0.97, f"[{mc}] notes carry the script value")
            if mc == "nomem":
                expect(all(r["report"] is None for r in rows), f"[{mc}] no report without memory")
                continue
            expect((rate([c for r in rows for c in r["report"]["faithful"]]) or 0) >= 0.97, f"[{mc}] report repeats step answers")

            forks, skipped = R.make_forks(ctx, eps)
            n_inj = sum(len(t["injections"]) for t in tasks)
            expect(len(skipped) == 0 and len(forks) == 3 * n_inj,
                   f"[{mc}] {len(forks)} forks = 3 arms x {n_inj} injection points, {len(skipped)} skipped")
            hist_k = [f for f in forks if f["arm"] == "inject"][0]
            expect(hist_k["records"][hist_k["k"] - 1]["edited"] == "inject", f"[{mc}] inject fork carries the edited record")
            R.run_forks(ctx, forks)
            by_task = {ep["task"]: ep for ep in eps}
            frows = [R.grade_fork(ctx, f, by_task[f["task"]], "mock") for f in forks]
            forks_all += frows

            def moved_steps(arm):
                return [s for r in frows if r["arm"] == arm for s in r["steps"] if s["after_k"] and s["moved"]]
            prop = rate([s["match_cf"] for s in moved_steps("inject")])
            expect(prop is not None and prop >= 0.97, f"[{mc}] inject: moved descendants equal the counterfactual ({f3(prop)})")
            for arm in ("sham", "null"):
                ok = rate([s["correct"] for s in moved_steps(arm)])
                expect(ok is not None and ok >= 0.97, f"[{mc}] {arm}: descendants stay correct ({f3(ok)})")
            spill = rate([s["correct"] for r in frows if r["arm"] == "inject"
                          for s in r["steps"] if s["after_k"] and not s["desc"]])
            expect(spill is None or spill >= 0.97, f"[{mc}] inject: non-descendants stay correct ({spill})")
            drift = rate([s["drift"] for r in frows if r["arm"] == "null" for s in r["steps"] if s["after_k"]])
            expect(drift == 0, f"[{mc}] null forks reproduce the clean run exactly (drift {drift})")
            lit = [s["literal"] for r in frows if r["arm"] == "inject" for s in r["steps"]
                   if s["after_k"] and s["child"] and s["literal"] is not None]
            expect(lit and all(lit), f"[{mc}] direct children copy the injected value as a literal ({sum(lit)}/{len(lit)})")
            expect(all(r["report"]["k_is_w"] for r in frows if r["arm"] == "inject"),
                   f"[{mc}] inject: the report repeats the injected value at step k")
            expect(all(r["report"]["k_is_truth"] for r in frows if r["arm"] == "sham"),
                   f"[{mc}] sham: the report keeps the correct value at step k")

        log("4. analysis on the self-test outputs")
        text, res = analyse(clean_all, forks_all, B=200)
        for (mdl, mc), m in sorted(res.items()):
            if mc == "nomem":
                continue
            d = m.get("d_wrong")
            expect(d is not None and d >= 0.97, f"[{mc}] analysis: inject - sham 'wrong' = {d}")
            pc = m.get("prop_c")
            expect(pc is not None and pc >= 0.97, f"[{mc}] analysis: P(inject = cf | sham correct) = {pc}")
        write_jsonl_gz(out / f"selftest_clean_{args.taskset}_{'-'.join(args.memconds)}_n{args.n_tasks}.jsonl.gz", clean_all)
        write_jsonl_gz(out / f"selftest_forks_{args.taskset}_{'-'.join(args.memconds)}_n{args.n_tasks}.jsonl.gz", forks_all)
        rep = out / f"selftest_analysis_{args.taskset}_{'-'.join(args.memconds)}_n{args.n_tasks}.txt"
        rep.write_text(text)
        log(f"wrote {rep.resolve()}")
    finally:
        ex.close()

    if FAILS:
        print(f"\nSELFTEST FAILED: {len(FAILS)} check(s)")
        for f in FAILS:
            print("  -", f)
        raise SystemExit(1)
    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
