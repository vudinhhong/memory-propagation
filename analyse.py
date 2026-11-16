"""Read the finished jobs and print the tables (no GPU).

    python analyse.py --taskset v2                      # every model and memory condition found
    python analyse.py --taskset v2real --models q07 q14 --memconds notes fullctx

Only files named exactly clean_/forks_<taskset>_<agent>_<model>_<memcond>.jsonl.gz in
<results_dir>/<model>/ are read (agent = config.json agent_version), so smoke and mock outputs
(_first<n>, _mock) are never mixed in. The report is written to
<results_dir>/analysis_<taskset>_<agent>_<models>_<memconds>.txt. An existing report is never
overwritten; the new one gets a time stamp.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))

from memprop.analysis import analyse                                    # noqa: E402
from memprop.common import ROOT, load_config, log, read_jsonl_gz, run_name  # noqa: E402


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskset", required=True, choices=sorted(cfg["tasksets"]))
    ap.add_argument("--models", nargs="+", default=sorted(cfg["models"]), choices=sorted(cfg["models"]))
    ap.add_argument("--memconds", nargs="+", default=cfg["memconds"], choices=cfg["memconds"])
    args = ap.parse_args()
    ts = args.taskset
    res_dir = ROOT / cfg["tasksets"][ts]["results_dir"]
    clean, forks, found, missing = [], [], [], []
    for m in args.models:
        for c in args.memconds:
            name = run_name(ts, cfg["agent_version"], m, c)
            cp = res_dir / m / f"clean_{name}.jsonl.gz"
            fp = res_dir / m / f"forks_{name}.jsonl.gz"
            if cp.exists() and fp.exists():
                clean += read_jsonl_gz(cp)
                forks += read_jsonl_gz(fp)
                found.append(f"{m}_{c}")
            else:
                missing.append(f"{m}_{c}")
    if not found:
        raise SystemExit(f"no finished runs under {res_dir}")
    print(f"finished: {' '.join(found)}")
    if missing:
        print(f"missing : {' '.join(missing)}")
    text, _ = analyse(clean, forks, B=cfg["bootstrap"])
    header = f"task set {ts}, agent {cfg['agent_version']}; runs: {' '.join(found)}\n"
    out = res_dir / f"analysis_{ts}_{cfg['agent_version']}_{'-'.join(args.models)}_{'-'.join(args.memconds)}.txt"
    if out.exists():
        out = out.with_name(f"{out.stem}_{time.strftime('%Y%m%d-%H%M%S')}{out.suffix}")
    out.write_text(header + text)
    print(header + text)
    log(f"wrote {out.resolve()}")


if __name__ == "__main__":
    main()
