"""Clean episodes, forks, and grading for one (model, memory condition).

Design of a fork:
  * the clean episode runs all S steps once, greedy decoding;
  * for every injection point k of the task at which the clean answer is CORRECT, three forks copy
    the clean history up to and including step k and regenerate steps k+1..S:
        null    record k untouched                      -> measures regeneration noise alone
        sham    record k rewritten, agent's own value  -> format change, same number
        inject  record k rewritten, wrong value w      -> the only difference from sham is the number
  * the effect of the injection is inject - sham, task by task.
Forks that start from an incorrect clean step k are not run (counted in the summary), because
"inject a wrong value" is not defined when the record is already wrong.
"""
import copy
import time

from . import agent as A
from .common import close, fmt, log, numbers_in, parse_answer, same
from .tasks import evaluate, task_df

ARMS = ("null", "sham", "inject")


class Ctx:
    def __init__(self, cfg, taskset, memcond, backend, executor, tasks):
        self.cfg = cfg
        self.taskset = taskset
        self.memcond = memcond
        self.backend = backend
        self.ex = executor
        self.tasks = {t["id"]: t for t in tasks}
        self._df = {}

    def df(self, task_id):
        if task_id not in self._df:
            self._df[task_id] = task_df(self.tasks[task_id])
        return self._df[task_id]

    def tol(self):
        return self.cfg["rtol"], self.cfg["atol"]


# ----------------------------------------------------------------------------- rounds

def _record(code, res, attempt, too_long):
    answer = parse_answer(res["stdout"]) if res["ok"] else None
    return {"code": code, "stdout": res["stdout"], "ok": bool(res["ok"] and answer is not None),
            "answer": answer, "error": res["error"], "attempts": attempt, "too_long": too_long,
            "note": None, "note_value": None, "edited": None}


def step_round(ctx, items):
    """items: list of (episode, j). Writes episode['records'][j-1] for each."""
    if not items:
        return
    cfg = ctx.cfg
    states = []
    for ep, j in items:
        conv = A.code_messages(ctx.tasks[ep["task"]], ctx.memcond, ep["records"], j, cfg)
        states.append({"ep": ep, "j": j, "conv": conv, "rec": None})
    pending = states
    for attempt in range(1, cfg["max_attempts"] + 1):
        replies = ctx.backend.chat([s["conv"] for s in pending], cfg["max_tokens_code"],
                                   metas=[{"kind": "code", "task": s["ep"]["task"], "j": s["j"]} for s in pending])
        codes = [A.extract_code(r) for r in replies]
        results = ctx.ex.run_many(codes)
        retry = []
        for s, reply, code, res in zip(pending, replies, codes, results):
            too_long = reply is None
            if too_long:
                res = {"ok": False, "stdout": "", "stderr": "", "error": "the prompt does not fit the context window"}
            s["rec"] = _record(code, res, attempt, too_long)
            if not s["rec"]["ok"] and not too_long and attempt < cfg["max_attempts"]:
                s["conv"] = A.retry_messages(s["conv"], reply, res["error"], res["stdout"])
                retry.append(s)
        pending = retry
        if not pending:
            break
    for s in states:
        s["ep"]["records"][s["j"] - 1] = s["rec"]
    if ctx.memcond == "notes":
        note_round(ctx, states)


def note_round(ctx, states):
    need = []
    for s in states:
        rec = s["rec"]
        if rec["ok"]:
            need.append(s)
        else:
            rec["note"] = f"Step {s['j']}: failed, no result"
    if not need:
        return
    convs = [A.note_messages(ctx.tasks[s["ep"]["task"]], s["j"], s["rec"]["stdout"], ctx.cfg) for s in need]
    replies = ctx.backend.chat(convs, ctx.cfg["max_tokens_note"],
                               metas=[{"kind": "note", "task": s["ep"]["task"], "j": s["j"]} for s in need])
    for s, r in zip(need, replies):
        line, v = A.parse_note(r)
        s["rec"]["note"] = line or f"Step {s['j']}: (empty note)"
        s["rec"]["note_value"] = v


def report_round(ctx, eps):
    if ctx.memcond == "nomem" or not eps:
        return
    convs = [A.report_messages(ctx.tasks[ep["task"]], ctx.memcond, ep["records"], ctx.cfg) for ep in eps]
    replies = ctx.backend.chat(convs, ctx.cfg["max_tokens_report"],
                               metas=[{"kind": "report", "task": ep["task"]} for ep in eps])
    for ep, r in zip(eps, replies):
        S = len(ctx.tasks[ep["task"]]["steps"])
        ep["report"] = {"text": (r or "")[:3000], "values": A.parse_report(r, S), "too_long": r is None}


# ----------------------------------------------------------------------------- phases

def run_clean(ctx, tasks):
    S = len(tasks[0]["steps"])
    eps = [{"task": t["id"], "arm": "clean", "k": None, "shown": None, "records": [None] * len(t["steps"]),
            "report": None} for t in tasks]
    for j in range(1, S + 1):
        t0 = time.time()
        step_round(ctx, [(ep, j) for ep in eps])
        ok = sum(ep["records"][j - 1]["ok"] for ep in eps)
        log(f"clean step {j}/{S}: {ok}/{len(eps)} produced an answer ({time.time() - t0:.0f}s)")
    report_round(ctx, eps)
    return eps


def make_forks(ctx, clean_eps):
    rtol, atol = ctx.tol()
    forks, skipped = [], []
    for ep in clean_eps:
        task = ctx.tasks[ep["task"]]
        S = len(task["steps"])
        for inj in task["injections"]:
            k = inj["k"]
            rec = ep["records"][k - 1]
            if not close(rec["answer"], task["truth"][k - 1], rtol, atol):
                skipped.append({"task": ep["task"], "k": k, "answer": rec["answer"]})
                continue
            is_int = task["steps"][k - 1]["is_int"]
            for arm in ARMS:
                f = {"task": ep["task"], "arm": arm, "k": k, "shown": None,
                     "records": [copy.deepcopy(r) for r in ep["records"][:k]] + [None] * (S - k), "report": None}
                if arm != "null":
                    text = fmt(rec["answer"] if arm == "sham" else inj["w"], is_int)
                    r = f["records"][k - 1]
                    r.update(stdout=A.edited_stdout(text), ok=True, error="", note=A.edited_note(k, text),
                             note_value=float(text), edited=arm)
                    f["shown"] = float(text)
                forks.append(f)
    return forks, skipped


def run_forks(ctx, forks):
    if not forks:
        return
    S = max(len(ctx.tasks[f["task"]]["steps"]) for f in forks)
    for j in range(2, S + 1):
        items = [(f, j) for f in forks if f["k"] < j]
        if not items:
            continue
        t0 = time.time()
        step_round(ctx, items)
        ok = sum(f["records"][j - 1]["ok"] for f, _ in items)
        log(f"forks step {j}/{S}: {ok}/{len(items)} produced an answer ({time.time() - t0:.0f}s)")
    report_round(ctx, forks)


# ----------------------------------------------------------------------------- grading

def _trim(rec, n):
    out = dict(rec)
    for key in ("stdout", "error"):
        v = out.get(key) or ""
        if len(v) > n:
            out[key] = v[:n // 2] + "\n...[trimmed]...\n" + v[-(n // 2):]
    return out


def _literal(code, shown, is_int):
    """Does the script contain the shown value of step k as a number literal (i.e. it was copied)?"""
    if shown is None or not code:
        return None
    if is_int and abs(shown) < 10:
        return None
    return any(abs(x - shown) <= max(5e-5, 1e-5 * abs(shown)) for x in numbers_in(code))


def grade_clean(ctx, ep, model_key):
    task = ctx.tasks[ep["task"]]
    rtol, atol = ctx.tol()
    steps = []
    for j, rec in enumerate(ep["records"], 1):
        ans = rec["answer"]
        steps.append({"j": j, "answer": ans, "ok": rec["ok"], "attempts": rec["attempts"], "too_long": rec["too_long"],
                      "correct": close(ans, task["truth"][j - 1], rtol, atol),
                      "note_value": rec["note_value"],
                      "note_ok": (close(rec["note_value"], ans, rtol, atol) if (ctx.memcond == "notes" and rec["ok"])
                                  else None)})
    rep = None
    if ep["report"] is not None:
        vals = ep["report"]["values"]
        rep = {"values": vals, "too_long": ep["report"]["too_long"],
               "correct": [close(v, t, rtol, atol) for v, t in zip(vals, task["truth"])],
               "faithful": [close(v, r["answer"], rtol, atol) for v, r in zip(vals, ep["records"])]}
    return {"taskset": ctx.taskset, "model": model_key, "memcond": ctx.memcond, "task": ep["task"],
            "dataset": task.get("dataset"), "arm": "clean", "k": None, "steps": steps, "report": rep, "report_text": (ep["report"] or {}).get("text"),
            "records": ep["records"]}                       # full records: forks are rebuilt from them on resume


def grade_fork(ctx, f, clean_ep, model_key):
    task = ctx.tasks[f["task"]]
    rtol, atol = ctx.tol()
    k = f["k"]
    inj = next(i for i in task["injections"] if i["k"] == k)
    is_int_k = task["steps"][k - 1]["is_int"]
    truth, cf = task["truth"], inj["cf"]
    clean_k = clean_ep["records"][k - 1]["answer"]
    # the correct chain when step k carries the 4-decimal value an agent reads in a sham record
    ref_round = evaluate(task["steps"], ctx.df(f["task"]), override={k: float(fmt(clean_k, is_int_k))})
    desc, moved = set(inj["descendants"]), set(inj["moved"])
    children = set(inj["children"])

    def ok_vs_truth(v, j):
        return close(v, truth[j - 1], rtol, atol) or close(v, ref_round[j - 1], rtol, atol)

    steps = []
    for j, rec in enumerate(f["records"], 1):
        ans = rec["answer"]
        row = {"j": j, "after_k": j > k, "answer": ans, "ok": rec["ok"], "attempts": rec["attempts"],
               "too_long": rec["too_long"], "desc": j in desc, "moved": j in moved, "child": j in children}
        if j > k:
            row["correct"] = ok_vs_truth(ans, j)
            row["match_cf"] = close(ans, cf[j - 1], rtol, atol) if j in moved else None
            row["drift"] = (not same(ans, clean_ep["records"][j - 1]["answer"], rtol, atol)) if f["arm"] == "null" else None
            row["literal"] = _literal(rec["code"], f["shown"], is_int_k) if j in desc else None
            row["note_ok"] = (close(rec["note_value"], ans, rtol, atol)
                              if (ctx.memcond == "notes" and rec["ok"]) else None)
        steps.append(row)
    rep = None
    if f["report"] is not None:
        vals = f["report"]["values"]
        rep = {"values": vals, "too_long": f["report"]["too_long"],
               "correct": [ok_vs_truth(v, j) for j, v in enumerate(vals, 1)],
               "match_cf": [close(v, cf[j - 1], rtol, atol) if j in moved else None for j, v in enumerate(vals, 1)],
               "k_is_w": close(vals[k - 1], inj["w"], rtol, atol),
               "k_is_truth": close(vals[k - 1], truth[k - 1], rtol, atol) or close(vals[k - 1], ref_round[k - 1], rtol, atol)}
    n = ctx.cfg["stdout_kept_chars"]
    return {"taskset": ctx.taskset, "model": model_key, "memcond": ctx.memcond, "task": f["task"],
            "dataset": task.get("dataset"), "arm": f["arm"], "k": k, "shown": f["shown"], "w": inj["w"], "depth_k": task["depth"][k - 1],
            "n_desc": len(desc), "n_moved": len(moved), "steps": steps, "report": rep,
            "report_text": (f["report"] or {}).get("text"),
            "records": [_trim(r, n) for r in f["records"][k:]]}     # regenerated steps only


def episodes_from_clean_rows(rows):
    return [{"task": r["task"], "arm": "clean", "k": None, "shown": None, "records": r["records"], "report": None}
            for r in rows]
