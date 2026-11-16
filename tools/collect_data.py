#!/usr/bin/env python3
"""Pack the data and run files of the release (run on the cluster, standard library only).

Copy this file into the experiment folder on the cluster (next to config.json) and run

    python collect_data.py                    # -> mem-propagation-data.tar.gz
    python collect_data.py --allow-missing    # pack what exists, list what does not

The archive holds exactly the files the release ships, at the paths the code expects, so it is
unpacked at the root of the release:

    data/tasks_<taskset>.json          task definitions, true values, counterfactual chains
    data/tasks_v1/, data/tasks_v2/     the 240 synthetic tables (CSV)
    data/real/<name>.json              metadata and SHA-256 of the six public tables (the tables
                                       themselves are rebuilt by code/prepare_real.py)
    results_<taskset>_p2/<model>/      clean_, forks_ and summary_ files of every run
    paper/figures/                     the figures and figdata_*.json from make_figures.py
    requirements-cluster.txt           pip freeze of the Python that ran this script

Only file names that match a real run exactly are packed: smoke and mock outputs (_first<n>, _mock)
and the invalid first agent version (p1: results_v1/, results_v1real/) are left out, as are the
job logs and the raw downloads under data/real/raw/.
"""
import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GITHUB_LIMIT = 95 * 1024 ** 2          # GitHub refuses files over 100 MB


def expected_files(cfg):
    files, optional = [], []
    agent = cfg["agent_version"]
    for ts, tcfg in cfg["tasksets"].items():
        files.append(f"data/tasks_{ts}.json")
        if tcfg["kind"] == "synthetic":
            csv_dir = HERE / "data" / f"tasks_{ts}"
            csvs = sorted(p.relative_to(HERE).as_posix() for p in csv_dir.glob("t*.csv"))
            if len(csvs) != tcfg["n_tasks"]:
                print(f"!! {csv_dir}: {len(csvs)} tables, config says {tcfg['n_tasks']}")
            files += csvs
        else:
            files += [f"data/real/{name}.json" for name in tcfg["datasets"]]
        res = tcfg["results_dir"]
        for m in cfg["models"]:
            for c in cfg["memconds"]:
                name = f"{ts}_{agent}_{m}_{c}"
                files += [f"{res}/{m}/clean_{name}.jsonl.gz", f"{res}/{m}/forks_{name}.jsonl.gz",
                          f"{res}/{m}/summary_{name}.json"]
    fig_dir = HERE / "paper" / "figures"
    if fig_dir.is_dir():
        optional += sorted(p.relative_to(HERE).as_posix() for p in fig_dir.iterdir()
                           if p.suffix in (".pdf", ".png", ".json"))
    return list(dict.fromkeys(files)), optional


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--out", default="mem-propagation-data.tar.gz")
    args = ap.parse_args()
    cfg = json.loads((HERE / "config.json").read_text())
    files, optional = expected_files(cfg)
    missing = [f for f in files if not (HERE / f).is_file()]
    if missing:
        print(f"{len(missing)} expected files are missing:")
        print("\n".join(f"  {f}" for f in missing[:40]))
        if not args.allow_missing:
            raise SystemExit("nothing packed (use --allow-missing to pack what exists)")
    present = [f for f in files if (HERE / f).is_file()] + optional
    if not any(f.startswith("paper/figures/") for f in optional):
        print("note: paper/figures/ not found -- run make_figures.py first if the figures belong in the archive")

    req = HERE / "requirements-cluster.txt"
    frz = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    req.write_text(f"# pip freeze of {sys.executable} (Python {sys.version.split()[0]})\n" + frz.stdout)
    present.append(req.name)

    sizes = {f: (HERE / f).stat().st_size for f in present}
    out = HERE / args.out
    with tarfile.open(out, "w:gz") as tar:
        for f in present:
            tar.add(HERE / f, arcname=f)
    total = sum(sizes.values())
    print(f"packed {len(present)} files, {total / 1024 ** 2:.1f} MB before compression -> {out} "
          f"({out.stat().st_size / 1024 ** 2:.1f} MB)")
    print("largest files:")
    for f, s in sorted(sizes.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {s / 1024 ** 2:8.1f} MB  {f}")
    big = [f for f, s in sizes.items() if s > GITHUB_LIMIT]
    if big:
        print(f"!! {len(big)} files exceed 95 MB and cannot be pushed to GitHub as ordinary files:")
        print("\n".join(f"  {f}" for f in big))


if __name__ == "__main__":
    main()
