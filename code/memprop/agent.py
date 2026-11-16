"""Prompts, history rendering and reply parsing for the step-by-step analysis agent.

The four memory conditions differ ONLY in what render_history() puts between the task header and
the current instruction:

  nomem    nothing -- earlier results must be recomputed from the plan
  notes    one line per earlier step, written by the model itself from the script output
  rawlog   the printed output of each earlier script
  fullctx  the script and its printed output for each earlier step

The record of step k that a fork edits is the thing the condition shows: the note (notes) or the
printed output (rawlog, fullctx). The sham and the injected record are written by the same
template and differ only in the number.
"""
import re

from .common import NUM, last_number, parse_answer, strip_thousands, to_float

# Agent version p2: the exact final line is stated and one example is shown. The earlier version p1
# asked only for a line "in the form ANSWER: <number>", which small models often wrote as code that
# prints nothing (see agent_version_note in config.json).
FINAL_LINE = 'print(f"ANSWER: {result}")'
SYSTEM_CODE = (
    "You are a careful data analyst working in Python. Only pandas and numpy are installed. You carry "
    "out a multi-step analysis one step at a time. For each step you write one complete, self-contained "
    "Python script that loads the data file, computes the requested number, stores it in a variable "
    "named `result`, and ends with exactly this line:\n"
    f"{FINAL_LINE}\n"
    "Reply with a single ```python code block and nothing else.\n\n"
    "Example of the expected form (a different file and question):\n"
    "```python\n"
    "import pandas as pd\n"
    "df = pd.read_csv(\"example.csv\")\n"
    "result = df[\"price\"].dropna().mean()\n"
    f"{FINAL_LINE}\n"
    "```")
SYSTEM_NOTE = "You keep a short notebook of results while carrying out a data analysis."
SYSTEM_REPORT = "You are a careful data analyst writing up the results of an analysis."

HIST_BEGIN = "=== History ==="
HIST_END = "=== End of history ==="
HIST_TITLE = {
    "notes": "Your notebook from the previous steps:",
    "rawlog": "Output printed by your scripts in the previous steps:",
    "fullctx": "Your previous steps (the script and its printed output):",
}

_FENCE = re.compile(r"```[ \t]*(?:python|py|Python|python3)?[ \t]*\n(.*?)(?:```|\Z)", re.S)


def _mid(text, n):
    text = text or ""
    if len(text) <= n:
        return text
    return text[:n // 2] + "\n...[truncated]...\n" + text[-(n // 2):]


def task_header(task):
    # A short relative file name; the executor copies the table into the scripts' working directory.
    lines = [f"Data file: {task['csv_name']} (in the current working directory)", f"Rows: {task['n_rows']}", "Columns:", *task["schema"], "",
             "First rows of the file:", task["head"].strip(), "",
             f"The analysis has {len(task['steps'])} steps:"]
    lines += [f"{i}. {s['text']}" for i, s in enumerate(task["steps"], 1)]
    return "\n".join(lines)


def shown_output(rec, cfg):
    if rec["ok"]:
        return _mid(rec["stdout"].rstrip(), cfg["stdout_shown_chars"])
    msg = (rec.get("error") or "").strip().splitlines()
    tail = msg[-1] if msg else "no result"
    return _mid(((rec["stdout"] or "").rstrip() + "\n" if rec.get("stdout") else "")
                + f"(the script did not produce a result: {tail})", cfg["stdout_shown_chars"])


def render_history(memcond, records, upto, cfg):
    # No history block before step 1 in any condition, so step 1 sees the same prompt everywhere.
    if memcond == "nomem" or upto == 0:
        return ""
    parts = []
    for i in range(1, upto + 1):
        r = records[i - 1]
        if memcond == "notes":
            parts.append(f"[Step {i}] {r.get('note') or f'Step {i}: (no note)'}")
        elif memcond == "rawlog":
            parts.append(f"[Step {i}] output:\n{shown_output(r, cfg)}")
        elif memcond == "fullctx":
            code = _mid((r.get("code") or "(no script)").rstrip(), cfg["code_shown_chars"])
            parts.append(f"[Step {i}] code:\n```python\n{code}\n```\n[Step {i}] output:\n{shown_output(r, cfg)}")
        else:
            raise ValueError(memcond)
    body = "\n".join(parts)
    return f"{HIST_TITLE[memcond]}\n{HIST_BEGIN}\n{body}\n{HIST_END}"


def code_messages(task, memcond, records, j, cfg):
    hist = render_history(memcond, records, j - 1, cfg)
    step = task["steps"][j - 1]
    extra = ("\nResults of earlier steps are not available to you; recompute anything you need."
             if memcond == "nomem" and step["refs"] else "")
    user = (task_header(task) + "\n\n" + (hist + "\n\n" if hist else "")
            + f"Now carry out step {j}: {step['text']}{extra}\n"
            f"Store the number in `result` and end the script with {FINAL_LINE}")
    return [{"role": "system", "content": SYSTEM_CODE}, {"role": "user", "content": user}]


def retry_messages(conv, reply, error, stdout):
    if error:
        why = "The script failed:\n" + _mid(error.strip(), 1200)
    else:
        why = ("The script ran but did not print a line `ANSWER: <number>`. It printed:\n"
               + _mid((stdout or "(nothing)").strip(), 600)
               + "\n`ANSWER: ...` written as a line of code prints nothing; it has to be printed.")
    return conv + [{"role": "assistant", "content": reply or ""},
                   {"role": "user", "content": why + f"\nFix it: store the number in `result`, end the script with "
                    f"{FINAL_LINE}, and reply with a single corrected ```python code block."}]


def note_messages(task, j, stdout, cfg):
    step = task["steps"][j - 1]
    user = (f"Step {j} of the analysis: {step['text']}\nYour script printed:\n"
            f"{_mid(stdout.rstrip(), cfg['stdout_shown_chars'])}\n\n"
            f"Write one line for your notebook in exactly this form:\nStep {j}: <short description> = <value>\n"
            "Use the value printed by the script.")
    return [{"role": "system", "content": SYSTEM_NOTE}, {"role": "user", "content": user}]


def report_messages(task, memcond, records, cfg):
    S = len(task["steps"])
    hist = render_history(memcond, records, S, cfg)
    lines = "\n".join(f"Step {i}: <number>" for i in range(1, S + 1))
    user = (task_header(task) + "\n\n" + hist + "\n\n"
            "Write the final report: one line per step, in exactly this form and nothing else:\n"
            f"{lines}\nUse the results recorded above.")
    return [{"role": "system", "content": SYSTEM_REPORT}, {"role": "user", "content": user}]


def edited_note(k, value_text):
    return f"Step {k}: result = {value_text}"


def edited_stdout(value_text):
    return f"ANSWER: {value_text}"


# ----------------------------------------------------------------------------- parsing

def extract_code(text):
    if not text:
        return None
    blocks = [b for b in _FENCE.findall(text) if b.strip()]
    if blocks:
        with_answer = [b for b in blocks if "ANSWER" in b]
        return (with_answer or blocks)[0]
    return text if ("print" in text and "import" in text) else None


def parse_note(text):
    """(note line, value). The value is the number after the last '=' on the note line."""
    if not text:
        return None, None
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    line = next((l for l in lines if "=" in l), lines[0] if lines else "")
    line = line.strip("`* ")[:300]
    value = None
    if "=" in line:
        m = re.search(NUM, strip_thousands(line.rsplit("=", 1)[1]))
        value = to_float(m.group(0)) if m else None
    if value is None:
        value = last_number(line)
    return line, value


_REPORT_LINE = re.compile(r"Step\s*(\d+)\s*[:=\-]\s*(.*)", re.I)


def parse_report(text, n_steps):
    vals = [None] * n_steps
    for line in (text or "").splitlines():
        m = _REPORT_LINE.search(line.replace("*", ""))
        if not m:
            continue
        i = int(m.group(1))
        if 1 <= i <= n_steps and vals[i - 1] is None:
            vals[i - 1] = last_number(m.group(2))
    return vals


def history_values(user_text):
    """Values an agent can read for each earlier step from a rendered history (used by the mock agent)."""
    if HIST_BEGIN not in user_text:
        return {}
    region = user_text.split(HIST_BEGIN, 1)[1].split(HIST_END, 1)[0]
    vals = {}
    for m in re.finditer(r"\[Step (\d+)\] output:\n(.*?)(?=\n\[Step \d+\]|\Z)", region, re.S):
        v = parse_answer(m.group(2))
        if v is not None:
            vals[int(m.group(1))] = v
    for m in re.finditer(r"^\[Step (\d+)\] (?!output:|code:)(.*)$", region, re.M):
        _, v = parse_note(m.group(2))
        if v is not None:
            vals[int(m.group(1))] = v
    return vals
