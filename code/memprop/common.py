"""Small helpers shared by every module: config, logging, number parsing, tolerant comparison, file output."""
import gzip
import json
import math
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # code/memprop/common.py -> project dir

NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
_NUM_RE = re.compile(NUM)
# The whole line must be `ANSWER: <number>`: printing a pandas Series gives "ANSWER: 0    3.21", which a
# looser pattern would read as 0. Such a line is treated as no answer, so the agent is asked to retry.
_ANSWER_RE = re.compile(r"\s*ANSWER\s*[:=]\s*(" + NUM + r")\s*")


def load_config():
    return json.loads((ROOT / "config.json").read_text())


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fmt(v, is_int):
    """The one way a value is written into an agent's history (sham and injected records alike)."""
    return str(int(round(v))) if is_int else f"{v:.4f}"


def close(a, b, rtol, atol):
    if a is None or b is None:
        return False
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    return abs(a - b) <= max(atol, rtol * abs(b))


def same(a, b, rtol, atol):
    """Two agent answers agree (both missing counts as agreeing)."""
    if a is None and b is None:
        return True
    return close(a, b, rtol, atol)


def strip_thousands(text):
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)


def to_float(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_answer(stdout):
    """Value on the last line of the output that reads `ANSWER: <number>`; None if there is none."""
    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        m = _ANSWER_RE.fullmatch(strip_thousands(line))
        if m:
            return to_float(m.group(1))
    return None


def last_number(text):
    if not text:
        return None
    nums = _NUM_RE.findall(strip_thousands(text))
    return to_float(nums[-1]) if nums else None


def numbers_in(text):
    if not text:
        return []
    out = []
    for s in _NUM_RE.findall(text):
        v = to_float(s)
        if v is not None:
            out.append(v)
    return out


def write_jsonl_gz(path, rows):
    """Write through a temporary file and os.replace, so a killed job never leaves half a file under the real name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    tmp = path.with_name(path.name + ".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
    log(f"wrote {path.resolve()} ({len(rows)} rows){' -- REPLACED an existing file' if existed else ''}")


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)
    log(f"wrote {path.resolve()}{' -- REPLACED an existing file' if existed else ''}")


def run_name(taskset, agent, model_key, memcond):
    """Name of one (task set, agent version, model, memory condition): every output file carries it."""
    return f"{taskset}_{agent}_{model_key}_{memcond}"


def job_name(taskset, agent, model_key):
    """Name of one job = one model on one task set, all memory conditions."""
    return f"{taskset}_{agent}_{model_key}"
