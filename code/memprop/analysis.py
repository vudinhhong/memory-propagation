"""Task-level aggregation and bootstrap intervals.

The unit of analysis is the TASK. Everything inside a task -- its injection points k, the steps
after k -- is averaged first, so a task with five injection points does not count five times.
Decoding is greedy and every configuration runs once, so there are no seeds to pool.

Tasks are independent only when each has its own table (synthetic task sets). In the real task
set, forty chains share each of six tables, so tasks are clustered by dataset: the bootstrap then
resamples datasets first and tasks within each drawn dataset second, and the interval reflects
that there are only six tables. Contrasts between arms, and between memory conditions of the same
model, are paired by task in both cases.
"""
from collections import defaultdict
from itertools import combinations

import numpy as np


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def boot(d, B, cluster_of=None, seed=0):
    """d: {task: value}. Returns (mean, lo, hi, n_tasks, n_clusters). Two-level when tasks share clusters."""
    items = [(t, v) for t, v in d.items() if v is not None]
    if not items:
        return None, None, None, 0, 0
    vals = np.array([v for _, v in items], float)
    m = float(vals.mean())
    groups = defaultdict(list)
    for t, v in items:
        groups[(cluster_of or {}).get(t) or t].append(v)
    G = len(groups)
    if len(items) == 1:
        return m, None, None, len(items), G
    rng = np.random.default_rng(seed)
    if G == len(items):
        means = vals[rng.integers(0, len(vals), size=(B, len(vals)))].mean(axis=1)
    else:
        arrs = [np.array(g, float) for g in groups.values()]
        means = np.empty(B)
        for b in range(B):
            parts = [arrs[g][rng.integers(0, len(arrs[g]), size=len(arrs[g]))] for g in rng.integers(0, G, size=G)]
            means[b] = np.concatenate(parts).mean()
    return m, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(items), G


def _fmt(b, pct=True):
    m, lo, hi, n, g = b
    if m is None:
        return "n/a"
    s = 100.0 if pct else 1.0
    size = f"n={n}" if g == n else f"n={n}, {g} datasets"
    if lo is None:
        return f"{m * s:6.1f}                 ({size})"
    return f"{m * s:6.1f} [{lo * s:6.1f}, {hi * s:6.1f}] ({size})"


# ----------------------------------------------------------------------------- per task

def clean_task_metrics(rows):
    out = {}
    for r in rows:
        st = r["steps"]
        m = {"acc": _mean([float(s["correct"]) for s in st]),
             "fail": _mean([float(not s["ok"]) for s in st]),
             "too_long": _mean([float(s["too_long"]) for s in st]),
             "note_fid": _mean([None if s.get("note_ok") is None else float(s["note_ok"]) for s in st])}
        rep = r.get("report")
        m["report_acc"] = _mean([float(c) for c in rep["correct"]]) if rep else None
        m["report_faithful"] = _mean([float(c) for c in rep["faithful"]]) if rep else None
        out[r["task"]] = m
    return out


def _fork_metrics(r):
    after = [s for s in r["steps"] if s["after_k"]]
    moved = [s for s in after if s["moved"]]
    m = {
        "wrong": _mean([float(not s["correct"]) for s in moved]),
        "prop": _mean([float(s["match_cf"]) for s in moved]),
        "wrong_child": _mean([float(not s["correct"]) for s in moved if s["child"]]),
        "wrong_deep": _mean([float(not s["correct"]) for s in moved if not s["child"]]),
        "spill": _mean([float(not s["correct"]) for s in after if not s["desc"]]),
        "literal": _mean([None if s.get("literal") is None else float(s["literal"]) for s in after if s["desc"]]),
        "drift": _mean([None if s.get("drift") is None else float(s["drift"]) for s in after]),
        "fail": _mean([float(not s["ok"]) for s in after]),
    }
    rep = r.get("report")
    if rep:
        m["rep_k_is_w"] = float(rep["k_is_w"])
        m["rep_k_is_truth"] = float(rep["k_is_truth"])
        m["rep_wrong"] = _mean([float(not rep["correct"][s["j"] - 1]) for s in moved])
        m["rep_prop"] = _mean([None if rep["match_cf"][s["j"] - 1] is None else float(rep["match_cf"][s["j"] - 1])
                               for s in moved])
    return m


def fork_task_metrics(rows, depth_filter=None):
    """{task: {arm: {metric: value}}}, each value the mean over the task's forks of that arm."""
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if depth_filter is not None and not depth_filter(r["depth_k"]):
            continue
        for key, v in _fork_metrics(r).items():
            if v is not None:
                acc[r["task"]][r["arm"]][key].append(v)
    return {t: {a: {k: float(np.mean(v)) for k, v in ms.items()} for a, ms in arms.items()}
            for t, arms in acc.items()}


def paired(ftm, metric, a="inject", b="sham"):
    out = {}
    for t, arms in ftm.items():
        x, y = arms.get(a, {}).get(metric), arms.get(b, {}).get(metric)
        if x is not None and y is not None:
            out[t] = x - y
    return out


def arm_values(ftm, metric, arm):
    return {t: arms[arm][metric] for t, arms in ftm.items() if arm in arms and metric in arms[arm]}


def conditional_metrics(rows):
    """Propagation where it CAN show: the same descendant step (task, k, j) answered correctly in the sham fork.

    inject-sham 'wrong' is capped by 1 - sham 'wrong': a model that gets 80% of the descendants wrong
    anyway can show at most 20 points of effect. Conditioning on the sham answer being correct removes
    that ceiling. Returns {task: {"prop_c": P(inject = counterfactual | sham correct),
                                  "keep_c": P(inject still correct | sham correct)}},
    averaged over j inside a fork, then over the task's forks, like every other metric.
    """
    pair = defaultdict(dict)
    for r in rows:
        if r["arm"] in ("sham", "inject"):
            pair[(r["task"], r["k"])][r["arm"]] = r
    acc = defaultdict(lambda: defaultdict(list))
    for (task, _k), arms in pair.items():
        if "sham" not in arms or "inject" not in arms:
            continue
        sham = {s["j"]: s for s in arms["sham"]["steps"]}
        prop, keep = [], []
        for s in arms["inject"]["steps"]:
            if s["after_k"] and s["moved"] and sham.get(s["j"], {}).get("correct"):
                prop.append(float(bool(s["match_cf"])))
                keep.append(float(s["correct"]))
        if prop:
            acc[task]["prop_c"].append(float(np.mean(prop)))
            acc[task]["keep_c"].append(float(np.mean(keep)))
    return {t: {k: float(np.mean(v)) for k, v in m.items()} for t, m in acc.items()}


# ----------------------------------------------------------------------------- report

FORK_METRICS = (("wrong", "wrong"), ("prop", "prop (= counterfactual)"),
                ("wrong_child", "wrong, direct child"), ("wrong_deep", "wrong, deeper descendant"),
                ("spill", "wrong, non-descendant"), ("literal", "value copied as literal"),
                ("fail", "step without answer"), ("rep_k_is_w", "report: step k = w"),
                ("rep_k_is_truth", "report: step k correct"), ("rep_wrong", "report: descendant wrong"),
                ("rep_prop", "report: descendant = counterfactual"))


def analyse(clean_rows, fork_rows, B=2000):
    """Returns (text, results). results[(model, memcond)] holds the headline numbers (used by the self-test)."""
    by = defaultdict(lambda: {"clean": [], "forks": []})
    cluster = {}
    for r in clean_rows:
        by[(r["model"], r["memcond"])]["clean"].append(r)
        if r.get("dataset"):
            cluster[r["task"]] = r["dataset"]
    for r in fork_rows:
        by[(r["model"], r["memcond"])]["forks"].append(r)
        if r.get("dataset"):
            cluster[r["task"]] = r["dataset"]
    datasets = sorted(set(cluster.values()))
    bt = lambda d: boot(d, B, cluster)                     # noqa: E731
    lines, results = [], {}
    P = lines.append
    P("Numbers are percentages; [95% bootstrap interval]; n = tasks"
      + ("; resampled by dataset, then by task within dataset." if datasets else "."))
    P("")
    P("=" * 100)
    P("1. Clean runs: step accuracy, failure, note fidelity, report accuracy")
    P("=" * 100)
    for key in sorted(by):
        ctm = clean_task_metrics(by[key]["clean"])
        if not ctm:
            continue
        P(f"\n{key[0]} / {key[1]}")
        for metric, label in (("acc", "step accuracy"), ("fail", "step without answer"),
                              ("too_long", "prompt too long"), ("note_fid", "note = script output"),
                              ("report_acc", "report accuracy"), ("report_faithful", "report = step answer")):
            vals = {t: m[metric] for t, m in ctm.items()}
            if any(v is not None for v in vals.values()):
                P(f"  {label:28s} {_fmt(bt(vals))}")
        results.setdefault(key, {})["clean_acc"] = bt({t: m["acc"] for t, m in ctm.items()})[0]

    P("")
    P("=" * 100)
    P("2. Forks: effect of one wrong value in the record of step k on the steps that depend on it")
    P("   wrong = answer is not the correct value; prop = answer equals the counterfactual value")
    P("   (computed from the injected number). Measured on descendants that the injection moves.")
    P("=" * 100)
    for key in sorted(by):
        forks = by[key]["forks"]
        if not forks:
            continue
        ftm = fork_task_metrics(forks)
        n_arm = {a: sum(1 for r in forks if r["arm"] == a) for a in ("null", "sham", "inject")}
        P(f"\n{key[0]} / {key[1]}   forks: {n_arm}   tasks: {len(ftm)}")
        res = results.setdefault(key, {})
        for metric, label in FORK_METRICS:
            cells = []
            for arm in ("null", "sham", "inject"):
                m = bt(arm_values(ftm, metric, arm))[0]
                cells.append(f"{arm} {m * 100:5.1f}" if m is not None else f"{arm}   n/a")
            d = bt(paired(ftm, metric))
            P(f"  {label:34s} {'  '.join(cells)}   inject-sham {_fmt(d)}")
            res[f"d_{metric}"] = d[0]
            res[f"inject_{metric}"] = bt(arm_values(ftm, metric, "inject"))[0]
            res[f"sham_{metric}"] = bt(arm_values(ftm, metric, "sham"))[0]
        cm = conditional_metrics(forks)
        for metric, label in (("prop_c", "inject = cf | sham correct"), ("keep_c", "inject correct | sham correct")):
            b = bt({t: m[metric] for t, m in cm.items()})
            P(f"  {label:34s} {_fmt(b)}")
            res[metric] = b[0]
        drift = bt(arm_values(ftm, "drift", "null"))
        P(f"  {'null fork differs from clean run':34s} {_fmt(drift)}")
        res["null_drift"] = drift[0]
        for name, flt in (("k is a root step (depth 0)", lambda d: d == 0), ("k is a derived step", lambda d: d > 0)):
            sub = fork_task_metrics(forks, flt)
            P(f"  {'inject-sham wrong, ' + name:34s} {_fmt(bt(paired(sub, 'wrong')))}")

    P("")
    P("=" * 100)
    P("3. Memory conditions compared within a model (paired by task)")
    P("   left: difference of inject-sham 'wrong'; right: difference of P(inject = cf | sham correct)")
    P("=" * 100)
    for mdl in sorted({k[0] for k in by}):
        conds = sorted({k[1] for k in by if k[0] == mdl and by[k]["forks"]})
        eff = {c: paired(fork_task_metrics(by[(mdl, c)]["forks"]), "wrong") for c in conds}
        cpr = {c: {t: m["prop_c"] for t, m in conditional_metrics(by[(mdl, c)]["forks"]).items()} for c in conds}
        for a, b in combinations(conds, 2):
            common = sorted(set(eff[a]) & set(eff[b]))
            cc = sorted(set(cpr[a]) & set(cpr[b]))
            P(f"  {mdl}: {a} - {b:8s} {_fmt(bt({t: eff[a][t] - eff[b][t] for t in common}))}"
              f"   |   {_fmt(bt({t: cpr[a][t] - cpr[b][t] for t in cc}))}")

    if datasets:
        P("")
        P("=" * 100)
        P("4. Per dataset (tasks of one table; interval = bootstrap over its tasks only)")
        P("   clean = clean step accuracy; effect = inject-sham 'wrong'; prop|sham ok = P(inject = cf | sham correct)")
        P("=" * 100)
        for key in sorted(by):
            ctm = clean_task_metrics(by[key]["clean"])
            ftm = fork_task_metrics(by[key]["forks"]) if by[key]["forks"] else {}
            P(f"\n{key[0]} / {key[1]}")
            for ds in datasets:
                keep = lambda d: {t: v for t, v in d.items() if cluster.get(t) == ds}   # noqa: E731
                c = boot(keep({t: m["acc"] for t, m in ctm.items()}), B)
                line = f"  {ds:10s} clean {_fmt(c)}"
                if ftm:
                    cm = conditional_metrics(by[key]["forks"])
                    line += f"   effect {_fmt(boot(keep(paired(ftm, 'wrong')), B))}"
                    line += f"   prop|sham ok {_fmt(boot(keep({t: m['prop_c'] for t, m in cm.items()}), B))}"
                P(line)
    return "\n".join(lines) + "\n", results
