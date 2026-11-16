"""Run agent-written Python in a pool of worker processes.

Workers are started with the 'spawn' method, so they never inherit a CUDA context from the
parent, and they are created before vLLM is loaded. Each worker imports pandas once and then
executes scripts with exec() in a fresh namespace, which is about a hundred times cheaper than a
new interpreter per script. A script gets a wall-clock limit (SIGALRM, raised as a BaseException
so `except Exception` in agent code cannot swallow it) and an address-space limit, so one runaway
script cannot take the job down with it.
"""
import contextlib
import io
import multiprocessing as mp
import os
import shutil
import signal
import traceback
from pathlib import Path


class _Timeout(BaseException):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def _init(workdir, mem_bytes):
    os.environ["MPLBACKEND"] = "Agg"
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except Exception:
        pass
    import numpy  # noqa: F401  warm the imports agent scripts will use
    import pandas  # noqa: F401
    signal.signal(signal.SIGALRM, _on_alarm)


def _run(code, timeout, max_chars):
    out, err = io.StringIO(), io.StringIO()
    ok, error = True, ""
    signal.alarm(int(timeout))
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            exec(compile(code, "<script>", "exec"), {"__name__": "__main__"})
    except _Timeout:
        ok, error = False, f"TimeoutError: the script ran longer than {timeout} seconds"
    except BaseException:                                          # noqa: BLE001  SystemExit included
        ok = False
        error = traceback.format_exc(limit=4)
    finally:
        signal.alarm(0)
    stdout = out.getvalue()
    if len(stdout) > max_chars:
        stdout = stdout[:max_chars // 2] + "\n...[output truncated]...\n" + stdout[-max_chars // 2:]
    return {"ok": ok, "stdout": stdout, "stderr": err.getvalue()[-2000:], "error": error[-2000:]}


class Executor:
    def __init__(self, workers, timeout, mem_gb, workdir, max_chars=20000, links=None):
        # links = {name: absolute path}: the data files, COPIED (read-only) into the scripts' working
        # directory, so prompts can give a short relative name instead of a long absolute path, and an
        # agent script that writes to that name (df.to_csv(...)) can never touch the shared original.
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        for name, target in (links or {}).items():
            dest = workdir / name
            if dest.is_symlink():
                dest.unlink()
            elif dest.exists():
                os.chmod(dest, 0o644)
            shutil.copyfile(target, dest)
            os.chmod(dest, 0o444)
        # BLAS threads in the workers: one each, or a 6-worker pool on an 8-core allocation oversubscribes
        for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
            os.environ[v] = "1"
        self.timeout = timeout
        self.max_chars = max_chars
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(workers, initializer=_init, initargs=(str(workdir), int(mem_gb * 2 ** 30)),
                             maxtasksperchild=200)

    def run_many(self, codes):
        """codes: list of str or None. Returns one result dict per entry (None -> not executed)."""
        handles = [None if c is None else self.pool.apply_async(_run, (c, self.timeout, self.max_chars))
                   for c in codes]
        results = []
        for h in handles:
            if h is None:
                results.append({"ok": False, "stdout": "", "stderr": "",
                                "error": "no Python code block was found in the reply"})
                continue
            try:
                results.append(h.get(timeout=self.timeout + 60))
            except mp.TimeoutError:
                results.append({"ok": False, "stdout": "", "stderr": "",
                                "error": "the worker running the script did not return (killed or hung)"})
            except Exception as e:                                  # noqa: BLE001 unpicklable result etc.
                results.append({"ok": False, "stdout": "", "stderr": "", "error": f"executor error: {e!r}"})
        return results

    def close(self):
        self.pool.terminate()
        self.pool.join()
