"""Chat backends. VLLMChat runs the real models; MockChat is a scripted agent for the CPU self-test.

Both take a batch of conversations plus one meta dict per conversation and return one reply string
per conversation (None when the prompt does not fit the context window -- that is recorded as a
failed step, never silently truncated).
"""
import gc

from . import agent as A
from .common import parse_answer
from .tasks import INT_TYPES


class VLLMChat:
    name = "vllm"

    def __init__(self, model_id, max_model_len, seed, gpu_memory_utilization, tensor_parallel_size=1):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        self._SP = SamplingParams
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.llm = LLM(model=model_id, dtype="bfloat16", seed=seed, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, enable_prefix_caching=True,
                       tensor_parallel_size=tensor_parallel_size)
        self.max_model_len = max_model_len
        self.seed = seed
        self.n_too_long = 0

    def _encode(self, messages):
        # Token ids, not text: a chat template's text already carries BOS, and tokenizing it again
        # would add a second one (Llama 3).
        ids = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]

    def chat(self, convs, max_tokens, metas=None):
        out = [None] * len(convs)
        prompts, idx = [], []
        for i, m in enumerate(convs):
            ids = self._encode(m)
            if len(ids) + max_tokens > self.max_model_len:
                self.n_too_long += 1
                continue
            prompts.append({"prompt_token_ids": ids})
            idx.append(i)
        if prompts:
            sp = self._SP(temperature=0.0, max_tokens=max_tokens, seed=self.seed)
            res = self.llm.generate(prompts, sp, use_tqdm=False)
            for i, r in zip(idx, res):
                out[i] = r.outputs[0].text
        return out

    def close(self):
        del self.llm
        gc.collect()


# ----------------------------------------------------------------------------- scripted agent

def reference_code(step, csv_path, deps, row_print=True):
    """A correct script for one step, with the values of the steps it uses written in as literals."""
    d = [repr(float(x)) for x in deps]
    L = ["import numpy as np", "import pandas as pd", f"df = pd.read_csv({csv_path!r})"]
    if row_print:
        L.append('print("rows loaded:", len(df))')
    t = step["type"]
    f = step.get("filter")
    sub = f"df[df[{f[0]!r}] == {f[1]!r}]" if f else "df"
    if t == "agg":
        expr = {"mean": "s.mean()", "median": "s.median()", "q25": "s.quantile(0.25)",
                "q75": "s.quantile(0.75)", "std": "s.std(ddof=1)"}[step["fn"]]
        L += [f"s = {sub}[{step['col']!r}].dropna()", f"v = float({expr})"]
    elif t == "count_gt":
        L.append(f"v = int((df[{step['col']!r}] > {d[0]}).sum())")
    elif t == "share_gt":
        L += [f"s = {sub}[{step['col']!r}].dropna()", f"v = float((s > {d[0]}).mean())"]
    elif t == "cond_mean":
        L.append(f"v = float(df.loc[df[{step['x']!r}] > {d[0]}, {step['y']!r}].dropna().mean())")
    elif t == "predict":
        L += [f"dd = df[[{step['x']!r}, {step['y']!r}]].dropna()",
              f"b, a = np.polyfit(dd[{step['x']!r}].to_numpy(float), dd[{step['y']!r}].to_numpy(float), 1)",
              f"v = float(a + b * {d[0]})"]
    elif t == "count_between":
        L += [f"lo, hi = min({d[0]}, {d[1]}), max({d[0]}, {d[1]})", f"c = df[{step['col']!r}]",
              "v = int(((c > lo) & (c < hi)).sum())"]
    elif t == "diff":
        L.append(f"v = {d[0]} - {d[1]}")
    elif t == "pct_change":
        L.append(f"v = 100.0 * ({d[0]} - {d[1]}) / {d[1]}")
    elif t == "ratio":
        L.append(f"v = {d[0]} / {d[1]}")
    elif t == "zscore":
        L.append(f"v = float((df.loc[df['id'] == {step['row_id']}, {step['col']!r}].iloc[0] - {d[0]}) / {d[1]})")
    else:
        raise ValueError(t)
    L.append('print(f"ANSWER: {int(v)}")' if t in INT_TYPES else 'print(f"ANSWER: {v:.6f}")')
    return "\n".join(L) + "\n"


class MockChat:
    """A perfect analyst that always REUSES a value it can read in its history and recomputes it
    from the data otherwise. It therefore propagates every injected value by construction, which
    is what lets the self-test check the whole measurement chain end to end.

    It also exercises the failure paths: on the first attempt it writes a crashing script for
    (task 0, step 3) and an endless loop for (task 1, step 2); the retry must recover both.
    """
    name = "mock"

    def __init__(self, tasks):
        self.tasks = {t["id"]: t for t in tasks}
        self.order = {t["id"]: i for i, t in enumerate(tasks)}
        self.n_too_long = 0

    def chat(self, convs, max_tokens, metas=None):
        return [self._one(c, m) for c, m in zip(convs, metas)]

    def _one(self, conv, meta):
        task = self.tasks[meta["task"]]
        user0 = conv[1]["content"]
        kind = meta["kind"]
        if kind == "note":
            v = parse_answer(user0.split("Your script printed:\n", 1)[1])
            return f"Step {meta['j']}: result = {v}"
        if kind == "report":
            hv = A.history_values(user0)
            return "\n".join(f"Step {i}: {hv.get(i, 'unknown')}" for i in range(1, len(task["steps"]) + 1))
        j = meta["j"]
        first_attempt = len(conv) == 2
        pos = self.order[task["id"]]
        if first_attempt and pos == 0 and j == 3:
            return "```python\nraise ValueError('mock failure on the first attempt')\n```"
        if first_attempt and pos == 1 and j == 2:
            return "```python\nwhile True:\n    pass\n```"
        step = task["steps"][j - 1]
        hv = A.history_values(user0)
        deps = [hv[i] if i in hv else task["truth"][i - 1] for i in step["refs"]]
        return "```python\n" + reference_code(step, task["csv_name"], deps) + "```"

    def close(self):
        pass
