"""Analysis tasks (on synthetic or public tables) whose every intermediate value is known exactly.

A task is a table plus a chain of S steps. Each step asks for one number; later steps use the
results of earlier ones (a threshold, a denominator, a reference value), so the steps form a
dependency graph. Because the steps are generated here, the reference evaluator gives the exact
value of every step, and also the exact value every step would take if one earlier result were
replaced by a given wrong value (the counterfactual).
That is what turns "the error propagated" into a checkable statement: a downstream answer that
equals the counterfactual value was computed from the injected number.

Everything uses pandas exactly the way the step text describes it (missing values dropped,
Series.quantile default interpolation, sample sd with ddof=1), and the table is round-tripped
through CSV before any value is computed, so the reference sees the same floats an agent reads
back with pd.read_csv.
"""
import io
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .common import close, fmt

# name, distribution, a, b, clip, decimals   (normal: mean a, sd b; lognormal: exp(a + b*z))
NUMERIC_POOL = [
    ("age_years", "normal", 46.0, 13.0, (18, 95), 0),
    ("weight_kg", "normal", 74.0, 13.0, (38, 160), 1),
    ("systolic_bp", "normal", 126.0, 16.0, (85, 210), 0),
    ("dose_mg", "lognormal", 3.6, 0.45, None, 1),
    ("reaction_ms", "lognormal", 6.0, 0.25, None, 0),
    ("income_k", "lognormal", 3.9, 0.5, None, 2),
    ("yield_pct", "normal", 61.0, 11.0, (0, 100), 2),
    ("temperature_c", "normal", 21.5, 3.5, None, 2),
    ("duration_min", "lognormal", 3.3, 0.6, None, 1),
    ("test_score", "normal", 70.0, 12.0, (0, 100), 1),
    ("length_cm", "normal", 31.0, 5.5, (5, 80), 2),
    ("concentration", "lognormal", -1.0, 0.7, None, 4),
]
GROUP_POOL = [
    ("site", ["North", "South", "East", "West", "Central"]),
    ("batch", ["B1", "B2", "B3", "B4"]),
    ("arm", ["control", "low", "high"]),
    ("lab", ["LabA", "LabB", "LabC", "LabD"]),
]
N_NUMERIC = 5
LOC_FNS = ("mean", "median", "q25", "q75")
FN_TEXT = {"mean": "mean", "median": "median",
           "q25": "25th percentile (pandas Series.quantile(0.25), default linear interpolation)",
           "q75": "75th percentile (pandas Series.quantile(0.75), default linear interpolation)",
           "std": "sample standard deviation (ddof=1)"}
INT_TYPES = ("count_gt", "count_between")
TYPE_WEIGHTS = {"count_gt": 1.0, "share_gt": 1.0, "cond_mean": 1.5, "predict": 1.0,
                "count_between": 1.0, "diff": 0.7, "pct_change": 0.7, "ratio": 0.7, "zscore": 1.0}
MIN_LEVEL_ROWS = 60


# ----------------------------------------------------------------------------- data

def make_dataset(rng):
    n = int(rng.integers(400, 2501))
    gname, levels = GROUP_POOL[int(rng.integers(len(GROUP_POOL)))]
    probs = rng.dirichlet(np.full(len(levels), 4.0))
    g = rng.choice(len(levels), size=n, p=probs)
    shift = rng.normal(0.0, 0.4, size=len(levels))
    z = rng.normal(size=n)
    picked = sorted(int(i) for i in rng.choice(len(NUMERIC_POOL), size=N_NUMERIC, replace=False))
    data = {"id": np.arange(1, n + 1), gname: np.array(levels, dtype=object)[g]}
    numeric = []
    for ci in picked:
        name, dist, a, b, clip, dec = NUMERIC_POOL[ci]
        rho = float(rng.uniform(-0.7, 0.7))
        e = rho * z + math.sqrt(1.0 - rho * rho) * rng.normal(size=n) + shift[g]
        x = a + b * e if dist == "normal" else np.exp(a + b * e)
        if clip is not None:
            x = np.clip(x, clip[0], clip[1])
        data[name] = np.round(x, dec).astype(float)
        numeric.append(name)
    missing_col = numeric[int(rng.integers(len(numeric)))]
    data[missing_col][rng.random(n) < 0.04] = np.nan
    df = pd.DataFrame(data)
    text = df.to_csv(index=False)
    df = pd.read_csv(io.StringIO(text))        # the floats an agent will read back
    meta = {"group_col": gname, "numeric": numeric, "missing_col": missing_col, "n_rows": n,
            "levels": sorted(str(v) for v in df[gname].unique()),
            "big_levels": sorted(str(l) for l, c in df[gname].value_counts().items() if c >= MIN_LEVEL_ROWS),
            "sd": {c: float(df[c].std(ddof=1)) for c in numeric}}
    return df, text, meta


# ----------------------------------------------------------------------------- evaluation

def _subset(df, filt):
    return df if not filt else df[df[filt[0]] == filt[1]]


def _div(a, b):
    return a / b if b != 0 else float("nan")


def compute_step(step, df, vals):
    """Value of one step, given the values of all earlier steps (vals[i-1] is step i)."""
    t = step["type"]
    r = [vals[i - 1] for i in step["refs"]]
    if t == "agg":
        s = _subset(df, step.get("filter"))[step["col"]].dropna()
        fn = step["fn"]
        if fn == "mean":
            return float(s.mean())
        if fn == "median":
            return float(s.median())
        if fn == "q25":
            return float(s.quantile(0.25))
        if fn == "q75":
            return float(s.quantile(0.75))
        if fn == "std":
            return float(s.std(ddof=1))
        raise ValueError(fn)
    if t == "count_gt":
        return float((df[step["col"]] > r[0]).sum())
    if t == "share_gt":
        s = _subset(df, step.get("filter"))[step["col"]].dropna()
        return float((s > r[0]).mean()) if len(s) else float("nan")
    if t == "cond_mean":
        return float(df.loc[df[step["x"]] > r[0], step["y"]].dropna().mean())
    if t == "predict":
        d = df[[step["x"], step["y"]]].dropna()
        b, a = np.polyfit(d[step["x"]].to_numpy(float), d[step["y"]].to_numpy(float), 1)
        return float(a + b * r[0])
    if t == "count_between":
        lo, hi = min(r), max(r)
        c = df[step["col"]]
        return float(((c > lo) & (c < hi)).sum())
    if t == "diff":
        return float(r[0] - r[1])
    if t == "pct_change":
        return float(100.0 * _div(r[0] - r[1], r[1]))
    if t == "ratio":
        return float(_div(r[0], r[1]))
    if t == "zscore":
        v = df.loc[df["id"] == step["row_id"], step["col"]].iloc[0]
        return float(_div(v - r[0], r[1]))
    raise ValueError(t)


def evaluate(steps, df, override=None):
    """Values of all steps in order; override = {k: value} fixes step k (1-based) to a given value."""
    override = override or {}
    vals = []
    for i, s in enumerate(steps, 1):
        if i in override:
            vals.append(float(override[i]))
            continue
        try:
            vals.append(compute_step(s, df, vals))
        except Exception:
            vals.append(float("nan"))
    return vals


# ----------------------------------------------------------------------------- step text

PREDICT_WORDINGS = ("ab", "plain")


def step_text(step, predict_wording="ab"):
    """predict_wording: how a `predict` step is phrased.
      "ab"    (task sets v1, v1real) names the line `y` = a + b * `x`; models that wrote
              `a, b = np.polyfit(...)` swapped the coefficients, since np.polyfit returns the slope first.
      "plain" (v2, v2real) names no coefficients. Nothing else about any step changes.
    """
    if predict_wording not in PREDICT_WORDINGS:
        raise ValueError(predict_wording)
    t = step["type"]
    f = step.get("filter")
    R = lambda i: f"the result of step {i}"                      # noqa: E731
    refs = step["refs"]
    if t == "agg":
        where = f" among rows where `{f[0]}` == '{f[1]}'" if f else ""
        return f"Compute the {FN_TEXT[step['fn']]} of `{step['col']}`{where}, ignoring missing values."
    if t == "count_gt":
        return (f"Count the rows where `{step['col']}` is strictly greater than {R(refs[0])}. "
                f"Rows where `{step['col']}` is missing are not counted.")
    if t == "share_gt":
        cond = ([f"`{f[0]}` == '{f[1]}'"] if f else []) + [f"`{step['col']}` is not missing"]
        return (f"Among the rows where {' and '.join(cond)}, compute the fraction (a number between 0 and 1) "
                f"whose `{step['col']}` is strictly greater than {R(refs[0])}.")
    if t == "cond_mean":
        return (f"Compute the mean of `{step['y']}`, ignoring missing values, over the rows where "
                f"`{step['x']}` is strictly greater than {R(refs[0])}.")
    if t == "predict":
        if predict_wording == "plain":
            return (f"Fit an ordinary least-squares straight line that predicts `{step['y']}` from `{step['x']}`, "
                    f"using the rows where both columns are present, and report the predicted value of "
                    f"`{step['y']}` when `{step['x']}` equals {R(refs[0])}.")
        return (f"Fit an ordinary least-squares line `{step['y']}` = a + b * `{step['x']}` on the rows where "
                f"both columns are present, and report the fitted value of `{step['y']}` at "
                f"`{step['x']}` equal to {R(refs[0])}.")
    if t == "count_between":
        return (f"Count the rows where `{step['col']}` lies strictly between {R(refs[0])} and {R(refs[1])} "
                f"(the smaller of the two is the lower bound). Rows where `{step['col']}` is missing are not counted.")
    if t == "diff":
        return f"Compute {R(refs[0])} minus {R(refs[1])}."
    if t == "pct_change":
        a, b = refs
        return (f"Compute the percentage change from {R(b)} to {R(a)}, that is "
                f"100 * (result of step {a} - result of step {b}) / result of step {b}.")
    if t == "ratio":
        return f"Compute {R(refs[0])} divided by {R(refs[1])}."
    if t == "zscore":
        return (f"Take the value of `{step['col']}` in the row with `id` == {step['row_id']}, subtract "
                f"{R(refs[0])}, and divide by {R(refs[1])}.")
    raise ValueError(t)


# ----------------------------------------------------------------------------- generation

def _filter(rng, meta, p):
    if meta["big_levels"] and rng.random() < p:
        return [meta["group_col"], meta["big_levels"][int(rng.integers(len(meta["big_levels"])))]]
    return None


def _agg(col, fn, filt):
    unit = ("scale:" if fn == "std" else "loc:") + col
    return {"type": "agg", "col": col, "fn": fn, "filter": filt, "refs": [], "unit": unit}


def _roots(meta, rng):
    num = meta["numeric"]
    x1 = num[int(rng.integers(len(num)))]
    fns = [LOC_FNS[int(i)] for i in rng.permutation(len(LOC_FNS))]
    roots = [_agg(x1, fns[0], _filter(rng, meta, 0.35)), _agg(x1, fns[1], _filter(rng, meta, 0.35))]
    if rng.random() < 0.5:
        roots.append(_agg(x1, "std", None))
    else:
        others = [c for c in num if c != x1]
        roots.append(_agg(others[int(rng.integers(len(others)))], fns[2], _filter(rng, meta, 0.35)))
    return roots


def _candidates(steps, df, meta, rng):
    locs, scales, counts = {}, {}, []
    for i, s in enumerate(steps, 1):
        u = s["unit"]
        if u.startswith("loc:"):
            locs.setdefault(u[4:], []).append(i)
        elif u.startswith("scale:"):
            scales.setdefault(u[6:], []).append(i)
        elif u == "count":
            counts.append(i)
    c = []
    for col, idx in locs.items():
        for a in idx:
            c.append({"type": "count_gt", "col": col, "refs": [a], "unit": "count"})
            c.append({"type": "share_gt", "col": col, "refs": [a], "filter": _filter(rng, meta, 0.4),
                      "unit": "share"})
            for y in meta["numeric"]:
                if y != col:
                    c.append({"type": "cond_mean", "y": y, "x": col, "refs": [a], "unit": "loc:" + y})
                    c.append({"type": "predict", "y": y, "x": col, "refs": [a], "unit": "loc:" + y})
            for b in scales.get(col, []):
                ok_ids = df.loc[df[col].notna(), "id"].to_numpy()
                rid = int(ok_ids[int(rng.integers(len(ok_ids)))])
                c.append({"type": "zscore", "col": col, "row_id": rid, "refs": [a, b], "unit": "free"})
        for a, b in combinations(idx, 2):
            p, q = (a, b) if rng.random() < 0.5 else (b, a)
            c.append({"type": "count_between", "col": col, "refs": [a, b], "unit": "count"})
            c.append({"type": "diff", "refs": [p, q], "unit": "free"})
            c.append({"type": "pct_change", "refs": [p, q], "unit": "free"})
    for a, b in combinations(counts, 2):
        p, q = (a, b) if rng.random() < 0.5 else (b, a)
        c.append({"type": "ratio", "refs": [p, q], "unit": "free"})
    return c


def _key(step):
    return json.dumps({k: v for k, v in step.items() if k not in ("text", "is_int")}, sort_keys=True)


def _pick(cands, steps, rng):
    last = len(steps)
    pool = cands
    if rng.random() < 0.6:                      # build depth: prefer steps that use the newest result
        recent = [c for c in cands if last in c["refs"]]
        if recent:
            pool = recent
    types = sorted({c["type"] for c in pool})
    w = np.array([TYPE_WEIGHTS[t] for t in types])
    t = types[int(rng.choice(len(types), p=w / w.sum()))]
    of_type = [c for c in pool if c["type"] == t]
    return of_type[int(rng.integers(len(of_type)))]


def _valid(step, value, df, vals):
    if value is None or not math.isfinite(value):
        return False
    n = len(df)
    t = step["type"]
    r = step["refs"]
    if t == "agg":
        if len(_subset(df, step.get("filter"))[step["col"]].dropna()) < 30:
            return False
        return step["fn"] != "std" or value > 0
    if t == "count_gt":
        return 0.05 * n <= value <= 0.95 * n
    if t == "count_between":
        return 0.03 * n <= value <= 0.95 * n
    if t == "share_gt":
        return 0.05 <= value <= 0.95
    if t == "cond_mean":
        m = (df[step["x"]] > vals[r[0] - 1]) & df[step["y"]].notna()
        return int(m.sum()) >= 30
    if t == "pct_change":
        return abs(vals[r[1] - 1]) > 1e-3 and 0.5 <= abs(value) < 1000
    if t == "ratio":
        return vals[r[1] - 1] >= 1
    if t == "zscore":
        return vals[r[1] - 1] > 0 and abs(value) < 8
    if t == "diff":
        return abs(value) > 1e-6
    return True


def graph(steps):
    S = len(steps)
    children = [[] for _ in range(S)]
    for j, s in enumerate(steps, 1):
        for i in s["refs"]:
            children[i - 1].append(j)
    desc = [set() for _ in range(S)]
    for i in range(S, 0, -1):
        for c in children[i - 1]:
            desc[i - 1].add(c)
            desc[i - 1] |= desc[c - 1]
    depth = []
    for s in steps:
        depth.append(0 if not s["refs"] else 1 + max(depth[i - 1] for i in s["refs"]))
    return children, [sorted(d) for d in desc], depth


def perturb(step, v, sign, meta):
    u = step["unit"]
    if u.startswith("loc:"):
        return v + sign * 0.5 * meta["sd"][u[4:]]
    if u.startswith("scale:"):
        return v * (1.0 + sign * 0.4)
    if u == "count":
        return float(max(1, round(v * (1.0 + sign * 0.25))))
    if u == "share":
        return float(min(0.99, max(0.01, v + sign * 0.15)))
    return v + sign * max(0.3 * abs(v), 0.5)


def injections(steps, df, vals, meta, rng, rtol, atol, moved_factor):
    """For every step with at least one dependent: a plausible wrong value w and the counterfactual chain.

    A step is eligible only if every DIRECT child moves by more than moved_factor tolerances when
    the step is set to w, so an injected value cannot vanish silently one step later. Deeper
    descendants that do not move are recorded and excluded from the propagation measure.
    """
    children, desc, _ = graph(steps)
    out = []
    for k in range(1, len(steps) + 1):
        if not children[k - 1]:
            continue
        step = steps[k - 1]
        is_int = step["type"] in INT_TYPES
        for sign in [int(s) for s in rng.permutation([-1, 1])]:
            w = float(fmt(perturb(step, vals[k - 1], sign, meta), is_int))
            if abs(w - vals[k - 1]) <= moved_factor * max(atol, rtol * abs(vals[k - 1])):
                continue
            cf = evaluate(steps, df, override={k: w})
            if not all(math.isfinite(cf[j - 1]) for j in desc[k - 1]):
                continue
            moved = [j for j in desc[k - 1]
                     if abs(cf[j - 1] - vals[j - 1]) > moved_factor * max(atol, rtol * abs(vals[j - 1]))]
            if not all(c in moved for c in children[k - 1]):
                continue
            out.append({"k": k, "w": w, "sign": sign, "cf": cf, "moved": moved,
                        "children": list(children[k - 1]), "descendants": list(desc[k - 1])})
            break
    return out


def _build_chain(df, meta, rng, n_steps, rtol, atol, moved_factor):
    """One attempt at a valid step chain on a given table; None if this attempt fails."""
    steps = _roots(meta, rng)
    vals = evaluate(steps, df)
    if not all(_valid(s, v, df, vals) for s, v in zip(steps, vals)):
        return None
    while len(steps) < n_steps:
        seen = {_key(s) for s in steps}
        cands = [c for c in _candidates(steps, df, meta, rng) if _key(c) not in seen]
        placed = False
        for _ in range(60):
            if not cands:
                break
            c = _pick(cands, steps, rng)
            try:
                v = compute_step(c, df, vals)
            except Exception:
                v = None
            if _valid(c, v, df, vals):
                steps.append(c)
                vals.append(v)
                placed = True
                break
            cands = [x for x in cands if x is not c]
        if not placed:
            return None
    children, desc, depth = graph(steps)
    if max(depth) < 2 or sum(1 for ch in children if ch) < 3:
        return None
    inj = injections(steps, df, vals, meta, rng, rtol, atol, moved_factor)
    if len(inj) < 2:
        return None
    return steps, vals, inj


def schema_lines(df, meta):
    out = []
    for c in df.columns:
        if c == "id":
            out.append(f"- id (integer): row identifier, 1 to {meta['n_rows']}")
        elif c == meta["group_col"]:
            out.append(f"- {c} (text): one of {', '.join(meta['levels'])}")
        elif pd.api.types.is_numeric_dtype(df[c]):
            out.append(f"- {c} (number{', has missing values' if df[c].isna().any() else ''})")
        else:
            out.append(f"- {c} (text)")
    return out


def _finish(task_id, seed, df, meta, steps, vals, inj, predict_wording):
    children, desc, depth = graph(steps)
    for s in steps:
        s["is_int"] = s["type"] in INT_TYPES
        s["text"] = step_text(s, predict_wording)
    return {"id": task_id, "seed": int(seed), "dataset": None, "n_rows": meta["n_rows"],
            "group_col": meta["group_col"], "levels": meta["levels"], "numeric": meta["numeric"],
            "missing_col": meta["missing_col"], "schema": schema_lines(df, meta),
            "head": df.head(3).to_csv(index=False),
            "steps": steps, "truth": vals, "children": children, "descendants": desc, "depth": depth,
            "injections": inj}


def chain_signature(steps):
    return json.dumps([_key(s) for s in steps])


def build_task(task_id, seed, n_steps, rtol, atol, moved_factor, predict_wording="ab"):
    """Synthetic task: a new table and a chain on it."""
    rng = np.random.default_rng(seed)
    for _attempt in range(500):
        df, csv_text, meta = make_dataset(rng)
        r = _build_chain(df, meta, rng, n_steps, rtol, atol, moved_factor)
        if r is not None:
            return _finish(task_id, seed, df, meta, *r, predict_wording), csv_text
    raise RuntimeError(f"could not build task {task_id} (seed {seed})")


def real_meta(df, info):
    """Meta for a prepared real table (data/real/<name>.json). Level-size floor scales with the table:
    60 rows as for synthetic tables, but no more than a fifth of the rows (Iris has 50 per species)."""
    g = info["group_col"]
    n = len(df)
    floor = min(MIN_LEVEL_ROWS, max(30, n // 5))
    counts = df[g].value_counts()
    return {"group_col": g, "numeric": list(info["numeric"]), "missing_col": None, "n_rows": n,
            "levels": sorted(str(v) for v in df[g].dropna().unique()),
            "big_levels": sorted(str(l) for l, c in counts.items() if c >= floor),
            "sd": {c: float(df[c].std(ddof=1)) for c in info["numeric"]}}


def build_task_real(task_id, seed, df, meta, dataset, n_steps, rtol, atol, moved_factor, avoid,
                    predict_wording="ab"):
    """Real-table task: the table is fixed, only the chain is drawn. `avoid` holds the signatures of
    chains already drawn on this table, so no two tasks of a dataset are the same chain."""
    rng = np.random.default_rng(seed)
    for _attempt in range(3000):
        r = _build_chain(df, meta, rng, n_steps, rtol, atol, moved_factor)
        if r is None:
            continue
        sig = chain_signature(r[0])
        if sig in avoid:
            continue
        task = _finish(task_id, seed, df, meta, *r, predict_wording)
        task["dataset"] = dataset
        return task, sig
    raise RuntimeError(f"could not build task {task_id} on {dataset} (seed {seed})")


def load_tasks(root, taskset):
    path = root / "data" / f"tasks_{taskset}.json"
    tasks = json.loads(path.read_text())["tasks"]
    for t in tasks:
        t["csv_abs"] = str((root / t["csv"]).resolve())
        t["csv_name"] = Path(t["csv"]).name          # what the agent sees; unique within a task set
    names = {}
    for t in tasks:
        if names.setdefault(t["csv_name"], t["csv_abs"]) != t["csv_abs"]:
            raise ValueError(f"two tables share the file name {t['csv_name']}")
    return tasks


def csv_links(tasks):
    """{short file name: absolute path} for Executor(links=...), which copies them read-only."""
    return {t["csv_name"]: t["csv_abs"] for t in tasks}


def task_df(task):
    return pd.read_csv(task["csv_abs"])


def check_task(task, rtol, atol):
    """Recompute the truth from the CSV on disk; returns a list of problems (empty = fine)."""
    df = task_df(task)
    vals = evaluate(task["steps"], df)
    probs = [f"{task['id']} step {i}: stored {a} recomputed {b}"
             for i, (a, b) in enumerate(zip(task["truth"], vals), 1) if not close(b, a, 1e-9, 1e-12)]
    for inj in task["injections"]:
        cf = evaluate(task["steps"], df, override={inj["k"]: inj["w"]})
        for j in inj["moved"]:
            if close(cf[j - 1], task["truth"][j - 1], rtol, atol):
                probs.append(f"{task['id']} inj k={inj['k']} step {j} does not move")
            if not close(cf[j - 1], inj["cf"][j - 1], 1e-9, 1e-12):
                probs.append(f"{task['id']} inj k={inj['k']} step {j}: stored cf differs")
    return probs
