#!/usr/bin/env python3
"""Draw every figure of the paper from the finished runs (no GPU; needs numpy, pandas, matplotlib).

    python make_figures.py                     # all figures -> paper/figures/
    python make_figures.py --only 5 6 S2       # some of them
    python make_figures.py --B 300             # quick bootstrap while checking the layout

Reads the same files as analyse.py (clean_/forks_<taskset>_<agent>_<model>_<memcond>.jsonl.gz for
the task sets v2 and v2real; v1 only for S2 Fig) and recomputes every number with the functions of
code/memprop/analysis.py, so a plotted value equals the value analyse.py prints. Each figure is
written as figN.pdf (for LaTeX) and figN.png (for checking); figdata_<figure>.json holds every
plotted value with its 95% bootstrap interval. Fig 3 also writes S2 Table to paper/S2_Table.tex
(the parent folder of --out). Existing figures are overwritten: they are derived files, rebuilt from
the run files at any time.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.legend_handler                                        # noqa: E402
import matplotlib.lines                                                 # noqa: E402
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch          # noqa: E402

from memprop.analysis import (arm_values, boot, clean_task_metrics, conditional_metrics,  # noqa: E402
                              fork_task_metrics, paired)
from memprop.common import ROOT, load_config, read_jsonl_gz, run_name  # noqa: E402
from memprop.tasks import load_tasks                                    # noqa: E402

MODELS = ["q01", "q03", "q07", "l08", "q14", "q32"]                    # by parameter count
MSHORT = {"q01": "Q-1.5B", "q03": "Q-3B", "q07": "Q-7B", "l08": "L-8B", "q14": "Q-14B", "q32": "Q-32B"}
CONDS = ["nomem", "notes", "rawlog", "fullctx"]
FCONDS = ["notes", "rawlog", "fullctx"]                                 # conditions with forks
CLABEL = {"nomem": "No memory", "notes": "Notes", "rawlog": "Raw log", "fullctx": "Full context"}
COLOR = {"nomem": "#8C8C8C", "notes": "#E69F00", "rawlog": "#0072B2", "fullctx": "#009E73"}  # Okabe-Ito
MARK = {"nomem": "D", "notes": "o", "rawlog": "s", "fullctx": "^"}
TSLABEL = {"v2": "Synthetic tables", "v2real": "Six public tables", "v1": "v1 wording"}
TSMARK = {"v2": "o", "v2real": "s"}
TSCOL = {"v2": "#0072B2", "v2real": "#D55E00"}                       # Figs 6 and 7
STEP_TYPES = ["count_gt", "share_gt", "cond_mean", "predict", "count_between", "diff", "pct_change",
              "ratio", "zscore"]
TYPE_LABEL = {"count_gt": "count >", "share_gt": "share >", "cond_mean": "cond. mean", "predict": "OLS predict",
              "count_between": "count between", "diff": "difference", "pct_change": "% change",
              "ratio": "ratio", "zscore": "z-score", "agg": "aggregate"}
FULL_W = 7.0          # inches; PLOS allows up to 7.5

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


# ----------------------------------------------------------------------------- data

class TaskSet:
    """All runs of one task set. Fork rows lose their stored scripts on loading (not needed here)."""

    def __init__(self, ts, cfg, B, forks=True):
        self.ts, self.B = ts, B
        self.tasks = {t["id"]: t for t in load_tasks(ROOT, ts)}
        res = ROOT / cfg["tasksets"][ts]["results_dir"]
        self.clean, self.forks, self.cluster = {}, {}, {}
        missing = []
        for m in MODELS:
            for c in CONDS:
                name = run_name(ts, cfg["agent_version"], m, c)
                cp, fp = res / m / f"clean_{name}.jsonl.gz", res / m / f"forks_{name}.jsonl.gz"
                if not cp.exists() or (forks and c != "nomem" and not fp.exists()):
                    missing.append(f"{m}_{c}")
                    continue
                rows = read_jsonl_gz(cp)
                for r in rows:
                    r.pop("report_text", None)
                    if r.get("dataset"):
                        self.cluster[r["task"]] = r["dataset"]
                self.clean[(m, c)] = rows
                if forks and c != "nomem":
                    frows = read_jsonl_gz(fp)
                    for r in frows:
                        r.pop("records", None)
                        r.pop("report_text", None)
                    self.forks[(m, c)] = frows
        if missing:
            raise SystemExit(f"{ts}: runs missing ({' '.join(missing)}); the figures need every model and condition")
        self._cache = {}
        print(f"loaded {ts}: {len(self.clean)} clean runs, {len(self.forks)} fork runs", flush=True)

    def _c(self, kind, key, fn):
        if (kind, key) not in self._cache:
            self._cache[(kind, key)] = fn()
        return self._cache[(kind, key)]

    def ctm(self, key):
        return self._c("ctm", key, lambda: clean_task_metrics(self.clean[key]))

    def ftm(self, key):
        return self._c("ftm", key, lambda: fork_task_metrics(self.forks[key]))

    def cm(self, key):
        return self._c("cm", key, lambda: conditional_metrics(self.forks[key]))

    def bt(self, d, clustered=True):
        """(mean, lo, hi, n) in percent; two-level bootstrap on a real task set."""
        m, lo, hi, n, _ = boot(d, self.B, self.cluster if (clustered and self.cluster) else None)
        f = lambda v: None if v is None else 100.0 * v                 # noqa: E731
        return {"mean": f(m), "lo": f(lo), "hi": f(hi), "n": n}

    # the quantities the figures use, each {task: value in [0, 1]}
    def acc(self, key):
        return {t: v["acc"] for t, v in self.ctm(key).items()}

    def prop_c(self, key):
        return {t: v["prop_c"] for t, v in self.cm(key).items()}

    def keep_c(self, key):
        return {t: v["keep_c"] for t, v in self.cm(key).items()}


def save(fig, name, outdir, data):
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight",
                    metadata={"CreationDate": None} if ext == "pdf" else None)   # byte-identical reruns
    plt.close(fig)
    (outdir / f"figdata_{name}.json").write_text(json.dumps(data, indent=1))
    print(f"  wrote {name}.pdf/.png and figdata_{name}.json", flush=True)


def top_legend(fig, handles, labels, ncol, y=1.0):
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False)


def panel_label(ax, s, x=-0.13, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def pointranges(ax, stats_by_cond, conds, width=0.6, connect=False, ms=4, strong=False):
    """stats_by_cond[c][i] = stat dict for model i. Draws dodged points with 95% intervals."""
    n = len(conds)
    offs = np.linspace(-width / 2, width / 2, n) if n > 1 else [0.0]
    for c, o in zip(conds, offs):
        xs, ms_, lo, hi = [], [], [], []
        for i, s in enumerate(stats_by_cond[c]):
            if s["mean"] is None:
                continue
            xs.append(i + o)
            ms_.append(s["mean"])
            lo.append(s["mean"] - (s["lo"] if s["lo"] is not None else s["mean"]))
            hi.append((s["hi"] if s["hi"] is not None else s["mean"]) - s["mean"])
        ax.errorbar(xs, ms_, yerr=[lo, hi], fmt=MARK[c], color=COLOR[c], ms=ms, lw=0.9, capsize=0,
                    label=CLABEL[c], zorder=3)
        if connect:
            ax.plot(xs, ms_, color=COLOR[c], lw=1.0 if strong else 0.6, alpha=0.85 if strong else 0.5, zorder=2)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MSHORT[m] for m in MODELS])
    ax.set_xlim(-0.6, len(MODELS) - 0.4)


# ----------------------------------------------------------------------------- Fig 1: design

def _box(ax, x, y, w, h, text, fc="white", ec="black", ls="-", fs=6.5, weight="normal", align="center", lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec,
                                lw=lw, ls=ls))
    tx = x + w / 2 if align == "center" else x + 0.6
    ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=fs, fontweight=weight, family="sans-serif")


def _arrow(ax, p, q, rad=0.0, color="black", lw=0.7, ls="-", head=6):
    ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=head,
                                 color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def fig_design(outdir, args):
    # one axes in tenths of an inch: x 0-70, y 0-56, so circles stay round and text sizes are predictable
    fig = plt.figure(figsize=(7.0, 5.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 56)
    ax.axis("off")
    red = "#C0392B"

    # A: a task as a dependency graph
    ax.text(0, 55.6, "A", fontsize=10, fontweight="bold", va="top")
    ax.text(2.6, 55.4, "One task: a table and eight dependent steps", fontsize=8, va="top")
    refs = {4: [1, 2], 5: [2], 6: [5], 7: [3, 2], 8: [6]}
    root = {1, 2, 3}
    xs = {i: 2.5 + (i - 1) * 4.0 for i in range(1, 9)}
    y0, r = 41.5, 1.2
    for j, rs in refs.items():
        for i in rs:
            _arrow(ax, (xs[i], y0 + r), (xs[j], y0 + r), rad=-0.5, color="#555555", lw=0.6)
    for i in range(1, 9):
        ax.add_patch(matplotlib.patches.Circle((xs[i], y0), r, fc="#EFEFEF" if i in root else "white",
                                               ec=red if i == 2 else "black", lw=1.6 if i == 2 else 0.7, zorder=3))
        ax.text(xs[i], y0, str(i), ha="center", va="center", fontsize=7, zorder=4)
    ax.text(xs[2], y0 - r - 0.4, "k", ha="center", va="top", fontsize=7.5, color=red, style="italic")
    for n, line in enumerate(["grey: computed from the table alone",
                              "arc i → j: step j uses the result of step i",
                              "descendants of k = 2: steps 4, 5, 6, 7 and 8"]):
        ax.text(0.8, 36.6 - n * 1.25, line, fontsize=6.2, va="top", color="#333333")

    # B: what each memory condition shows of an earlier step
    ax.text(35.5, 55.6, "B", fontsize=10, fontweight="bold", va="top")
    ax.text(38.1, 55.4, "What a later prompt shows of step 2", fontsize=8, va="top")
    rows = [("nomem", "nothing: earlier results must be recomputed"),
            ("notes", "[Step 2] Step 2: mean of weight_kg = 74.1832"),
            ("rawlog", "[Step 2] output:\nANSWER: 74.1832"),
            ("fullctx", "[Step 2] code: df = pd.read_csv(...) ...\n[Step 2] output: ANSWER: 74.1832")]
    for n, (c, txt) in enumerate(rows):
        y = 48.6 - n * 4.0
        _box(ax, 36.0, y, 8.6, 3.0, CLABEL[c], fc=COLOR[c], ec=COLOR[c], fs=6.8, weight="bold")
        _box(ax, 45.4, y, 24.2, 3.0, txt, fc="white", ec="#999999", fs=6.2, align="left")

    # C: forks
    ax.text(0, 32.6, "C", fontsize=10, fontweight="bold", va="top")
    ax.text(2.6, 32.4, "Forks at an injection point k = 2 (run only if the clean answer at step 2 is correct)",
            fontsize=8, va="top")
    bw, bh, step = 6.4, 2.6, 7.4
    x0 = 7.6
    bx = lambda i: x0 + (i - 1) * step                                  # noqa: E731
    yc = 26.4
    ax.text(0.3, yc + bh / 2, "clean run", fontsize=7, va="center", fontweight="bold")
    for i in range(1, 9):
        _box(ax, bx(i), yc, bw, bh, f"step {i}", fc="#EFEFEF" if i <= 2 else "white", fs=6.5)
    arms = [("null", None, "record of step 2 unchanged: measures regeneration noise"),
            ("sham", "74.1832", "record rewritten in the edit template with the agent's own value"),
            ("inject", "80.6000", "same template with a plausible wrong value w: the only difference from sham")]
    ys = [20.0, 14.0, 8.0]
    _arrow(ax, (bx(2) + bw / 2, yc), (bx(2) + bw / 2, ys[0] + bh + 0.1), color="#777777", lw=0.7)
    ax.text(bx(3), (yc + ys[0] + bh) / 2, "each arm copies steps 1-2 of the clean run, edits the record of step 2 "
            "and regenerates steps 3-8", fontsize=6.2, va="center", color="#333333")
    for (arm, val, desc), y in zip(arms, ys):
        col = red if arm == "inject" else "black"
        ax.text(0.3, y + bh / 2, arm, fontsize=7.5, va="center", fontweight="bold", color=col)
        _box(ax, bx(1), y, bw, bh, "step 1", fc="#EFEFEF", fs=6.5)
        if val is None:
            _box(ax, bx(2), y, bw, bh, "step 2", fc="#EFEFEF", fs=6.5)
        else:
            _box(ax, bx(2), y, bw, bh, f"step 2\n{val}", fc="#EFEFEF", ec=col, lw=1.4, fs=6.2)
        for i in range(3, 9):
            _box(ax, bx(i), y, bw, bh, f"step {i}", ls="--", fs=6.5)
        ax.text(bx(3), y - 0.75, desc, fontsize=6.2, va="top", color="#333333")
    _box(ax, 0.3, 0.3, 69.3, 4.2,
         "Scored on each descendant that w moves:  correct = equals the true value;  "
         "propagated = equals the counterfactual value computed from w\n"
         "Primary measure:  P(inject = counterfactual | same step correct in sham), averaged within tasks, "
         "bootstrap over tasks", fc="#FAFAFA", ec="#999999", fs=6.2)
    save(fig, "fig1", outdir, {"note": "schematic; example numbers are illustrative"})


# ----------------------------------------------------------------------------- Fig 2: clean accuracy

def fig_clean(sets, outdir, args):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), sharey=True)
    data = {}
    for ax, ts, lab in zip(axes, ("v2", "v2real"), "AB"):
        S = sets[ts]
        width = 0.2
        for ci, c in enumerate(CONDS):
            st = [S.bt(S.acc((m, c))) for m in MODELS]
            data.setdefault(ts, {})[c] = dict(zip(MODELS, st))
            x = np.arange(len(MODELS)) + (ci - 1.5) * width
            ax.bar(x, [s["mean"] for s in st], width * 0.92, color=COLOR[c], label=CLABEL[c], zorder=2)
            ax.errorbar(x, [s["mean"] for s in st], yerr=[[s["mean"] - s["lo"] for s in st], [s["hi"] - s["mean"] for s in st]],
                        fmt="none", ecolor="black", lw=0.6, zorder=3)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MSHORT[m] for m in MODELS])
        ax.set_ylim(0, 100)
        ax.set_title(f"{TSLABEL[ts]} (240 tasks)")
        ax.grid(axis="y", lw=0.3, alpha=0.5, zorder=0)
        panel_label(ax, lab)
    axes[0].set_ylabel("Clean-run step accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False)
    save(fig, "fig2", outdir, data)


# ----------------------------------------------------------------------------- Fig 3: accuracy by step

DEPTHS = [0, 1, 2, 3]                     # depth in the dependency graph; 3 stands for 3 or more
DEPTH_LABEL = ["0", "1", "2", "\u22653"]


def depth_accuracy(S, key):
    """{depth bin: stat}: per task, the share of its steps at that depth answered correctly; tasks
    without a step at that depth do not enter that bin."""
    out = {}
    for d in DEPTHS:
        vals = {}
        for r in S.clean[key]:
            depth = S.tasks[r["task"]]["depth"]
            xs = [float(st["correct"]) for st in r["steps"] if min(depth[st["j"] - 1], 3) == d]
            if xs:
                vals[r["task"]] = float(np.mean(xs))
        out[d] = S.bt(vals)
    return out


def position_at_depth1(S, key):
    """{"early", "late"}: per task, the share of its depth-1 steps answered correctly at positions 4-5
    (early) and 6-8 (late). Same difficulty, different length of the history shown to the agent."""
    out = {}
    for name, lo, hi in (("early", 4, 5), ("late", 6, 8)):
        vals = {}
        for r in S.clean[key]:
            depth = S.tasks[r["task"]]["depth"]
            xs = [float(st["correct"]) for st in r["steps"] if depth[st["j"] - 1] == 1 and lo <= st["j"] <= hi]
            if xs:
                vals[r["task"]] = float(np.mean(xs))
        out[name] = S.bt(vals)
    return out


def write_s2_table(sets, path):
    """S2 Table: (A) clean-run accuracy by depth; (B) depth-1 steps early and late in the chain."""
    acc = {ts: {(m, c): depth_accuracy(sets[ts], (m, c)) for m in MODELS for c in CONDS} for ts in ("v2", "v2real")}
    rows = []
    for m in MODELS:
        for i, c in enumerate(CONDS):
            cells = [f"{acc[ts][(m, c)][d]['mean']:.1f}" for ts in ("v2", "v2real") for d in DEPTHS]
            name = MSHORT[m].replace("Q-", "Qwen2.5-").replace("L-", "Llama-3.1-") if i == 0 else ""
            rows.append(f"{name} & {CLABEL[c].lower()} & " + " & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")
    ntask = {ts: {d: acc[ts][(MODELS[0], CONDS[0])][d]["n"] for d in DEPTHS} for ts in ("v2", "v2real")}
    n_line = " & ".join(str(ntask[ts][d]) for ts in ("v2", "v2real") for d in DEPTHS)
    pos = {ts: {(m, c): position_at_depth1(sets[ts], (m, c)) for m in MODELS for c in CONDS}
           for ts in ("v2", "v2real")}
    rows_b = []
    for m in MODELS:
        for i, c in enumerate(CONDS):
            cells = []
            for ts in ("v2", "v2real"):
                e, l = pos[ts][(m, c)]["early"]["mean"], pos[ts][(m, c)]["late"]["mean"]
                cells += [f"{e:.1f}", f"{l:.1f}", f"{l - e:+.1f}"]
            name = MSHORT[m].replace("Q-", "Qwen2.5-").replace("L-", "Llama-3.1-") if i == 0 else ""
            rows_b.append(f"{name} & {CLABEL[c].lower()} & " + " & ".join(cells) + r" \\")
        rows_b.append(r"\addlinespace")
    k0 = (MODELS[0], CONDS[0])
    nb = " & ".join(f"{pos[ts][k0]['early']['n']} & {pos[ts][k0]['late']['n']} & " for ts in ("v2", "v2real"))
    tex = r"""% Generated by make_figures.py (Fig 3) from the run files -- do not edit by hand.
\documentclass[10pt]{article}
\usepackage[margin=2cm,a4paper]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\pagestyle{empty}
\begin{document}
\noindent\textbf{S2 Table. Clean-run step accuracy by depth in the dependency graph, and by position at a fixed depth.}

\noindent\textbf{A. By depth.} Accuracy (\%) of the steps at each depth: depth 0 = computed from the table alone (always steps 1--3),
depth $d$ = the longest chain of earlier results the step depends on has $d$ links; $\geq$3 pools depths 3 to 5.
Values are means over tasks of the share of a task's steps at that depth answered correctly; a task enters a
column only if it has a step at that depth (number of tasks in the last row).

\vspace{1em}
\noindent\begin{tabular}{llrrrrrrrr}
\toprule
 & & \multicolumn{4}{c}{Synthetic tables} & \multicolumn{4}{c}{Six public tables} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
Model & Memory & 0 & 1 & 2 & $\geq$3 & 0 & 1 & 2 & $\geq$3 \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
\multicolumn{2}{l}{Tasks in the column} & """ + n_line + r""" \\
\bottomrule
\end{tabular}

\vspace{2em}
\noindent\textbf{B. Depth-1 steps early and late in the chain.} Accuracy (\%) of the steps at depth 1 that stand at
positions 4--5 (early) and 6--8 (late). The steps have the same depth, so they are equally far from the table, but
the history shown to the agent is longer for the late ones. $\Delta$ = late $-$ early, in points.

\vspace{1em}
\noindent\begin{tabular}{llrrrrrr}
\toprule
 & & \multicolumn{3}{c}{Synthetic tables} & \multicolumn{3}{c}{Six public tables} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Model & Memory & early & late & $\Delta$ & early & late & $\Delta$ \\
\midrule
""" + "\n".join(rows_b) + r"""
\midrule
\multicolumn{2}{l}{Tasks in the column} & """ + nb.rstrip(" &") + r""" & \\
\bottomrule
\end{tabular}
\end{document}
"""
    Path(path).write_text(tex)
    print(f"  wrote {path}", flush=True)
    for ts in ("v2", "v2real"):
        d = [pos[ts][k]["late"]["mean"] - pos[ts][k]["early"]["mean"] for k in pos[ts]]
        print(f"  {ts}: depth-1 late - early from {min(d):+.1f} to {max(d):+.1f} points, mean {np.mean(d):+.1f}",
              flush=True)
    return acc


def fig_by_step(sets, outdir, args):
    """Fig 3: clean-run accuracy by depth (synthetic tables); also writes S2 Table (both task sets)."""
    acc = write_s2_table(sets, Path(outdir).parent / "S2_Table.tex")["v2"]
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 4.3), sharex=True, sharey=True)
    data = {}
    for ax, m in zip(axes.flat, MODELS):
        for c in CONDS:
            st = [acc[(m, c)][d] for d in DEPTHS]
            data.setdefault(m, {})[c] = dict(zip(DEPTH_LABEL, st))
            ax.plot(DEPTHS, [s["mean"] for s in st], marker=MARK[c], ms=3.2, color=COLOR[c], label=CLABEL[c])
            ax.fill_between(DEPTHS, [s["lo"] for s in st], [s["hi"] for s in st], color=COLOR[c], alpha=0.12, lw=0)
        ax.set_title(MSHORT[m].replace("Q-", "Qwen2.5-").replace("L-", "Llama-3.1-"), pad=6)
        ax.set_ylim(-3, 105)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xticks(DEPTHS)
        ax.set_xticklabels(DEPTH_LABEL)
        ax.set_xlim(-0.2, 3.2)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    for ax in axes[1]:
        ax.set_xlabel("Depth in the dependency graph")
    for ax in axes[:, 0]:
        ax.set_ylabel("Step accuracy (%)")
    fig.tight_layout(h_pad=1.0)
    h, l = axes[0, 0].get_legend_handles_labels()
    top_legend(fig, h, l, 4, y=0.985)
    save(fig, "fig3", outdir, data)


# ----------------------------------------------------------------------------- Fig 4: controls

def fig_controls(sets, outdir, args):
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 4.6), sharex=True)
    data = {}
    specs = [("Null drift", "steps differing from clean run (%)",
              lambda S, k: arm_values(S.ftm(k), "drift", "null"), (0, 40)),
             ("Sham minus null", "descendants wrong (points)",
              lambda S, k: paired(S.ftm(k), "wrong", "sham", "null"), (-15, 15)),
             ("Spillover: inject minus sham", "non-descendants wrong (points)",
              lambda S, k: paired(S.ftm(k), "spill"), (-15, 15))]
    for row, ts in enumerate(("v2", "v2real")):
        S = sets[ts]
        for col, (title, ylab, fn, ylim) in enumerate(specs):
            ax = axes[row, col]
            st = {c: [S.bt(fn(S, (m, c))) for m in MODELS] for c in FCONDS}
            data.setdefault(ts, {})[title] = {c: dict(zip(MODELS, v)) for c, v in st.items()}
            pointranges(ax, st, FCONDS, ms=3.2)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_ylim(*ylim)
            ax.grid(axis="y", lw=0.3, alpha=0.5)
            if row == 0:
                ax.set_title(title, pad=8)
            ax.set_ylabel(ylab if col else f"{TSLABEL[ts]}\n{ylab}", fontsize=7)
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=6.5)
            panel_label(ax, "ABCDEF"[row * 3 + col], x=-0.25, y=1.04)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    fig.tight_layout(w_pad=1.5, h_pad=1.2)
    save(fig, "fig4", outdir, data)


# ----------------------------------------------------------------------------- Fig 5: headline

def fig_headline(sets, outdir, args):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.8), sharey=True)
    data = {}
    for ax, ts, lab in zip(axes, ("v2", "v2real"), "AB"):
        S = sets[ts]
        st = {c: [S.bt(S.prop_c((m, c))) for m in MODELS] for c in FCONDS}
        data[ts] = {c: dict(zip(MODELS, v)) for c, v in st.items()}
        pointranges(ax, st, FCONDS, connect=True)
        ax.set_ylim(-3, 103)
        ax.set_yticks(range(0, 101, 20))
        ax.set_title(f"{TSLABEL[ts]}")
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        panel_label(ax, lab)
    axes[0].set_ylabel("P(inject = counterfactual | sham correct) (%)")
    h, l = axes[0].get_legend_handles_labels()
    top_legend(fig, h, l, 3)
    save(fig, "fig5", outdir, data)


# ----------------------------------------------------------------------------- Fig 7: condition contrasts

def fig_contrasts(sets, outdir, args):
    """Fig 7: models on the x axis as in Figs 5 and 8; one shaded column per model holds its two points."""
    pairs = [("notes", "rawlog"), ("fullctx", "rawlog"), ("fullctx", "notes")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.9))
    data = {}
    tscol = TSCOL
    for ax, (a, b), lab in zip(axes, pairs, "ABC"):
        for i in range(len(MODELS)):
            if i % 2 == 0:
                ax.axvspan(i - 0.5, i + 0.5, color="#EEEEEE", lw=0, zorder=0)
        for oi, ts in enumerate(("v2", "v2real")):
            S = sets[ts]
            xs, ms_, lo, hi = [], [], [], []
            for i, m in enumerate(MODELS):
                pa, pb = S.prop_c((m, a)), S.prop_c((m, b))
                d = {t: pa[t] - pb[t] for t in sorted(set(pa) & set(pb))}
                s = S.bt(d)
                data.setdefault(f"{a}-{b}", {}).setdefault(ts, {})[m] = s
                xs.append(i + (-0.17 if oi == 0 else 0.17))
                ms_.append(s["mean"])
                lo.append(s["mean"] - s["lo"])
                hi.append(s["hi"] - s["mean"])
            ax.errorbar(xs, ms_, yerr=[lo, hi], fmt=TSMARK[ts], color=tscol[ts],
                        mfc=tscol[ts] if ts == "v2" else "white", ms=4, lw=1.0, capsize=0, label=TSLABEL[ts],
                        zorder=3)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{CLABEL[a]} minus {CLABEL[b].lower()}")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MSHORT[m] for m in MODELS])
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=6.5)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, len(MODELS) - 0.5)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        panel_label(ax, lab, x=-0.2)
    axes[0].set_ylabel("Difference in P(inject = cf | sham correct)\n(percentage points)")
    fig.tight_layout(w_pad=1.5)
    h, l = axes[0].get_legend_handles_labels()
    top_legend(fig, h, l, 2, y=0.98)
    save(fig, "fig7", outdir, data)


# ----------------------------------------------------------------------------- Fig 8: mechanism

def fig_mechanism(sets, outdir, args):
    S = sets["v2"]
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.3), sharex=True)
    specs = [("Recovery: inject still correct when sham is correct", "steps (%)", lambda k: S.keep_c(k), (0, 100)),
             ("Value of step k copied into the script (inject arm)", "scripts (%)",
              lambda k: arm_values(S.ftm(k), "literal", "inject"), (0, 100)),
             ("Copying, inject minus sham", "difference (points)", lambda k: paired(S.ftm(k), "literal"), (-10, 10)),
             ("Final report gives w for step k (inject arm)", "forks (%)",
              lambda k: arm_values(S.ftm(k), "rep_k_is_w", "inject"), (90, 100.8))]
    data = {}
    for ax, (title, ylab, fn, ylim), lab in zip(axes.flat, specs, "ABCD"):
        st = {c: [S.bt(fn((m, c))) for m in MODELS] for c in FCONDS}
        data[title] = {c: dict(zip(MODELS, v)) for c, v in st.items()}
        pointranges(ax, st, FCONDS, connect=True, ms=4, strong=True)
        ax.set_ylim(*ylim)
        if ylim[0] < 0:
            ax.axhline(0, color="black", lw=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        panel_label(ax, lab, x=-0.12)
    h, l = axes[0, 0].get_legend_handles_labels()
    top_legend(fig, h, l, 3)
    fig.tight_layout(h_pad=1.5)
    save(fig, "fig8", outdir, data)


# ----------------------------------------------------------------------------- Fig 6: competence vs trust

# where each model name goes in Fig 6 (data coordinates); a thin leader line joins it to the model's circle
COMP_LABELS = {
    "notes": {"q01": (33, 47), "q03": (61, 5), "q07": (87, 57), "l08": (52, 86), "q14": (76, 100), "q32": (86, 91)},
    "rawlog": {"q01": (33, 15), "q03": (44, 32), "q07": (68, 73), "l08": (70, 4), "q14": (83, 80), "q32": (78, 100)},
    "fullctx": {"q01": (48, 3), "q03": (67, 22), "q07": (80, 57), "l08": (66, 46), "q14": (84, 84), "q32": (83, 100)},
}


def fig_competence(sets, outdir, args):
    """One panel per memory condition. Each model has a filled circle (synthetic tables) and an open
    square (public tables) joined by a thin line, and is named next to its circle."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.9), sharex=True, sharey=True)
    data = []
    for ax, c, lab in zip(axes, FCONDS, "ABC"):
        for m in MODELS:
            pts = {}
            for ts in ("v2", "v2real"):
                S = sets[ts]
                x, y = S.bt(S.acc((m, c))), S.bt(S.prop_c((m, c)))
                pts[ts] = (x, y)
                data.append({"taskset": ts, "model": m, "memcond": c, "clean_acc": x, "prop_c": y})
                ax.errorbar(x["mean"], y["mean"], xerr=[[x["mean"] - x["lo"]], [x["hi"] - x["mean"]]],
                            yerr=[[y["mean"] - y["lo"]], [y["hi"] - y["mean"]]], fmt=TSMARK[ts], color=TSCOL[ts],
                            mfc=TSCOL[ts] if ts == "v2" else "white", ms=4, lw=0.5, capsize=0, zorder=3)
            (x1, y1), (x2, y2) = pts["v2"], pts["v2real"]
            ax.plot([x1["mean"], x2["mean"]], [y1["mean"], y2["mean"]], color="#888888", lw=0.6, zorder=2)
            ax.annotate(MSHORT[m], (x1["mean"], y1["mean"]), xytext=COMP_LABELS[c][m], textcoords="data",
                        fontsize=6, color="#222222", ha="center", va="center", zorder=4,
                        arrowprops=dict(arrowstyle="-", color="#555555", lw=0.5, shrinkA=4, shrinkB=3))
        ax.set_title(CLABEL[c])
        ax.set_xlim(30, 103)
        ax.set_ylim(-5, 105)
        ax.grid(lw=0.3, alpha=0.5)
        ax.set_xlabel("Clean-run step accuracy (%)")
        panel_label(ax, lab, x=-0.12)
    axes[0].set_ylabel("P(inject = cf | sham correct) (%)")
    fig.tight_layout(w_pad=1.0)
    h = [matplotlib.lines.Line2D([], [], ls="", marker="o", color=TSCOL["v2"], ms=4),
         matplotlib.lines.Line2D([], [], ls="", marker="s", color=TSCOL["v2real"], mfc="white", ms=4),
         matplotlib.lines.Line2D([], [], color="#888888", lw=0.6)]
    top_legend(fig, h, [TSLABEL["v2"], TSLABEL["v2real"], "same model on both"], 3, y=0.97)
    save(fig, "fig6", outdir, data)


# ----------------------------------------------------------------------------- Fig 9: depth

DEPTH_MARK = [("o", "o"), ("D", "D")]          # row 0: child / deeper; row 1: root k / derived k
DEPTH_COL = [("#0072B2", "#56B4E9"), ("#D55E00", "#E69F00")]   # one colour per series in every column


def fig_depth(sets, outdir, args):
    S = sets["v2"]
    fig, axes = plt.subplots(2, 3, figsize=(FULL_W, 4.0), sharex=True, sharey=True)
    data = {}
    for col, c in enumerate(FCONDS):
        sub_root = {m: fork_task_metrics(S.forks[(m, c)], lambda d: d == 0) for m in MODELS}
        sub_der = {m: fork_task_metrics(S.forks[(m, c)], lambda d: d > 0) for m in MODELS}
        rows = [
            ({"direct child": [S.bt(paired(S.ftm((m, c)), "wrong_child")) for m in MODELS],
              "deeper descendant": [S.bt(paired(S.ftm((m, c)), "wrong_deep")) for m in MODELS]}),
            ({"k is a root step": [S.bt(paired(sub_root[m], "wrong")) for m in MODELS],
              "k is a derived step": [S.bt(paired(sub_der[m], "wrong")) for m in MODELS]}),
        ]
        for row, series in enumerate(rows):
            ax = axes[row, col]
            for si, (name, st) in enumerate(series.items()):
                data.setdefault(c, {})[name] = dict(zip(MODELS, st))
                o = -0.13 if si == 0 else 0.13
                xs = [i + o for i in range(len(MODELS))]
                ax.errorbar(xs, [s["mean"] for s in st],
                            yerr=[[s["mean"] - s["lo"] for s in st], [s["hi"] - s["mean"] for s in st]],
                            fmt=DEPTH_MARK[row][si], color=DEPTH_COL[row][si],
                            mfc=DEPTH_COL[row][si] if si == 0 else "white",
                            ms=4, lw=0.8, capsize=0)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_ylim(-10, 100)
            ax.grid(axis="y", lw=0.3, alpha=0.5)
            ax.set_xticks(range(len(MODELS)))
            ax.set_xticklabels([MSHORT[m] for m in MODELS], fontsize=6.3)
            if row == 0:
                ax.set_title(CLABEL[c])
            if col == 0:
                ax.set_ylabel(("By descendant depth" if row == 0 else "By position of k")
                              + "\ninject - sham, wrong (points)")
            panel_label(ax, "ABCDEF"[row * 3 + col], x=-0.16)
    names = ["direct child", "deeper descendant", "k is a root step", "k is a derived step"]
    h = [matplotlib.lines.Line2D([], [], ls="", marker=DEPTH_MARK[i // 2][i % 2], color=DEPTH_COL[i // 2][i % 2],
                                 mfc=DEPTH_COL[i // 2][i % 2] if i % 2 == 0 else "white", ms=4) for i in range(4)]
    fig.tight_layout(h_pad=1.0)
    top_legend(fig, h, names, 4, y=0.985)
    save(fig, "fig9", outdir, data)


# ----------------------------------------------------------------------------- Fig 10: per dataset

def _heatmap(ax, mat, rows, cols, vmin=0, vmax=100, cmap="Blues", fmt="{:.0f}", fs=5.6):
    m = np.ma.masked_invalid(np.array(mat, float))
    im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = mat[i][j]
            if v is None or not np.isfinite(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs,
                    color="white" if (v - vmin) / (vmax - vmin) > 0.6 else "black")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    return im


def fig_datasets(sets, outdir, args):
    S = sets["v2real"]
    dss = sorted(set(S.cluster.values()))
    rows, mat, data = [], [], {}
    for m in MODELS:
        for c in FCONDS:
            pc = S.prop_c((m, c))
            line = []
            for ds in dss:
                d = {t: v for t, v in pc.items() if S.cluster.get(t) == ds}
                s = S.bt(d, clustered=False)
                data.setdefault(f"{m}_{c}", {})[ds] = s
                line.append(s["mean"])
            rows.append(f"{MSHORT[m]}  {CLABEL[c].lower()}")
            mat.append(line)
    fig, ax = plt.subplots(figsize=(4.2, 5.6))
    im = _heatmap(ax, mat, rows, dss)
    for r in range(3, len(rows), 3):
        ax.axhline(r - 0.5, color="white", lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("P(inject = counterfactual | sham correct) (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, "fig10", outdir, data)


# ----------------------------------------------------------------------------- S1 Fig: by step type

def fig_types(sets, outdir, args):
    S = sets["v2"]
    rows, mat, data = [], [], {}
    for m in MODELS:
        for c in FCONDS:
            pair = defaultdict(dict)
            for r in S.forks[(m, c)]:
                if r["arm"] in ("sham", "inject"):
                    pair[(r["task"], r["k"])][r["arm"]] = r
            acc = defaultdict(lambda: defaultdict(list))         # type -> task -> fork means
            for (task, _k), arms in pair.items():
                if "sham" not in arms or "inject" not in arms:
                    continue
                sham = {s["j"]: s for s in arms["sham"]["steps"]}
                per_type = defaultdict(list)
                for s in arms["inject"]["steps"]:
                    if s["after_k"] and s["moved"] and sham.get(s["j"], {}).get("correct"):
                        per_type[S.tasks[task]["steps"][s["j"] - 1]["type"]].append(float(bool(s["match_cf"])))
                for t, v in per_type.items():
                    acc[t][task].append(float(np.mean(v)))
            line = []
            for t in STEP_TYPES:
                d = {task: float(np.mean(v)) for task, v in acc[t].items()}
                s = S.bt(d) if len(d) >= 15 else {"mean": None, "lo": None, "hi": None, "n": len(d)}
                data.setdefault(f"{m}_{c}", {})[t] = s
                line.append(s["mean"] if s["mean"] is not None else np.nan)
            rows.append(f"{MSHORT[m]}  {CLABEL[c].lower()}")
            mat.append(line)
    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    im = _heatmap(ax, mat, rows, [TYPE_LABEL[t] for t in STEP_TYPES])
    for r in range(3, len(rows), 3):
        ax.axhline(r - 0.5, color="white", lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("P(inject = counterfactual | sham correct) (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, "S1_Fig", outdir, data)


# ----------------------------------------------------------------------------- S2 Fig: task wording

def _taxonomy(tracer, rows, memcond):
    """Wrong answers at predict steps of a clean run, by cause (the classification of diagnose.py --trace)."""
    cnt = Counter()
    for r in rows:
        task = tracer.tasks[r["task"]]
        rec_vals = [(rec.get("note_value") if memcond == "notes" else rec.get("answer")) for rec in r["records"]]
        for s in r["steps"]:
            step = task["steps"][s["j"] - 1]
            if step["type"] != "predict" or s["correct"] or not s["ok"]:
                continue
            ups = {i: rec_vals[i - 1] for i in step["refs"]}
            if any(v is None for v in ups.values()):
                cnt["upstream not recorded"] += 1
                continue
            vals = list(task["truth"])
            for i, v in ups.items():
                vals[i - 1] = v
            try:
                v = tracer.compute_step(step, tracer.df(r["task"]), vals)
            except Exception:                                   # noqa: BLE001
                v = None
            if v is not None and tracer.close(s["answer"], v, tracer.rtol, tracer.atol):
                cnt["from upstream value"] += 1
            elif tracer.swapped(step, r["task"], vals, s["answer"]):
                cnt["coefficients swapped"] += 1
            elif tracer.no_intercept(step, r["task"], vals, s["answer"]):
                cnt["no intercept"] += 1
            else:
                cnt["other"] += 1
    return cnt


def fig_wording(sets, cfg, outdir, args):
    sys.path.insert(0, str(HERE))
    from diagnose import Tracer                                  # noqa: E402
    v1 = TaskSet("v1", cfg, args.B, forks=False)
    v2 = sets["v2"]
    fig = plt.figure(figsize=(FULL_W, 2.9))
    axA = fig.add_axes([0.06, 0.18, 0.36, 0.72])
    data = {"predict_acc": {}, "taxonomy": {}}
    # A: accuracy at predict steps, clean runs, pooled over the four conditions
    for oi, (S, ts) in enumerate(((v1, "v1"), (v2, "v2"))):
        st = []
        for m in MODELS:
            d = defaultdict(list)
            for c in CONDS:
                for r in S.clean[(m, c)]:
                    for s in r["steps"]:
                        if S.tasks[r["task"]]["steps"][s["j"] - 1]["type"] == "predict":
                            d[r["task"]].append(float(s["correct"]))
            st.append(S.bt({t: float(np.mean(v)) for t, v in d.items()}))
        data["predict_acc"][ts] = dict(zip(MODELS, st))
        x = np.arange(len(MODELS)) + (oi - 0.5) * 0.36
        axA.bar(x, [s["mean"] for s in st], 0.34, color="#E69F00" if ts == "v1" else "#0072B2",
                label="v1: 'y = a + b * x'" if ts == "v1" else "v2: plain wording", zorder=2)
        axA.errorbar(x, [s["mean"] for s in st], yerr=[[s["mean"] - s["lo"] for s in st], [s["hi"] - s["mean"] for s in st]],
                     fmt="none", ecolor="black", lw=0.6, zorder=3)
    axA.set_xticks(range(len(MODELS)))
    axA.set_xticklabels([MSHORT[m] for m in MODELS], fontsize=6.3)
    axA.set_ylim(0, 100)
    axA.set_ylabel("Accuracy at OLS-predict steps (%)")
    axA.legend(loc="upper left", frameon=False, fontsize=6.3)
    axA.grid(axis="y", lw=0.3, alpha=0.5, zorder=0)
    panel_label(axA, "A", x=-0.16)
    # B, C: causes of wrong predict answers for the two largest models
    causes = ["coefficients swapped", "no intercept", "other", "from upstream value", "upstream not recorded"]
    ccol = {"coefficients swapped": "#D55E00", "no intercept": "#CC79A7", "other": "#999999",
            "from upstream value": "#56B4E9", "upstream not recorded": "#F0E442"}
    tr = {"v1": Tracer("v1"), "v2": Tracer("v2")}
    for pi, m in enumerate(("q14", "q32")):
        ax = fig.add_axes([0.5 + pi * 0.25, 0.18, 0.21, 0.72])
        labels, bottoms = [], np.zeros(6)
        cols = []
        for c in FCONDS:
            for ts, S in (("v1", v1), ("v2", v2)):
                cols.append((c, ts, _taxonomy(tr[ts], S.clean[(m, c)], c)))
                labels.append({"notes": "N", "rawlog": "R", "fullctx": "F"}[c] + ts[-1])
        for cause in causes:
            vals = np.array([cnt.get(cause, 0) for _, _, cnt in cols], float)
            ax.bar(range(6), vals, 0.7, bottom=bottoms, color=ccol[cause], label=cause, zorder=2)
            bottoms += vals
        for c, ts, cnt in cols:
            data["taxonomy"].setdefault(m, {}).setdefault(c, {})[ts] = dict(cnt)
        ax.set_ylim(0, bottoms.max() * 1.08)
        ax.set_xticks(range(6))
        ax.set_xticklabels(labels, fontsize=6.3)
        ax.set_title(MSHORT[m].replace("Q-", "Qwen2.5-"))
        ax.grid(axis="y", lw=0.3, alpha=0.5, zorder=0)
        if pi == 0:
            ax.set_ylabel("Wrong answers at predict steps")
        panel_label(ax, "BC"[pi], x=-0.2)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.73, 0.06), ncol=3, frameon=False, fontsize=6)
    fig.text(0.73, -0.09, "N/R/F = notes / raw log / full context; 1 = v1 wording, 2 = v2 wording", fontsize=6,
             ha="center")
    save(fig, "S2_Fig", outdir, data)


# ----------------------------------------------------------------------------- main

FIGS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "S1", "S2"]


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=FIGS, default=FIGS)
    ap.add_argument("--B", type=int, default=cfg["bootstrap"], help="bootstrap resamples (default from config)")
    ap.add_argument("--out", default=str(HERE / "paper" / "figures"))
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    need = set(args.only)
    sets = {}
    if need - {"1"}:
        sets["v2"] = TaskSet("v2", cfg, args.B)
    if need & {"2", "3", "4", "5", "6", "7", "10"}:
        sets["v2real"] = TaskSet("v2real", cfg, args.B)
    jobs = {"1": lambda: fig_design(outdir, args), "2": lambda: fig_clean(sets, outdir, args),
            "3": lambda: fig_by_step(sets, outdir, args), "4": lambda: fig_controls(sets, outdir, args),
            "5": lambda: fig_headline(sets, outdir, args), "6": lambda: fig_competence(sets, outdir, args),
            "7": lambda: fig_contrasts(sets, outdir, args), "8": lambda: fig_mechanism(sets, outdir, args),
            "9": lambda: fig_depth(sets, outdir, args), "10": lambda: fig_datasets(sets, outdir, args),
            "S1": lambda: fig_types(sets, outdir, args), "S2": lambda: fig_wording(sets, cfg, outdir, args)}
    for f in FIGS:
        if f in need:
            print(f"Fig {f}", flush=True)
            jobs[f]()
    print(f"figures in {outdir.resolve()}")


if __name__ == "__main__":
    main()
