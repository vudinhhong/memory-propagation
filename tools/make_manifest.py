#!/usr/bin/env python3
"""Write MANIFEST.sha256 for the release and check it is ready to push (standard library only).

    python tools/make_manifest.py            # write MANIFEST.sha256 at the release root
    python tools/make_manifest.py --check    # verify the files against an existing manifest

Before writing, it checks that the task definitions and run files are present, that no file exceeds
GitHub's 100 MB limit, and that no text file contains a pattern listed in tools/forbidden.txt (an
optional local list, one pattern per line, not distributed).
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".DS_Store"}
SKIP_FILES = {"MANIFEST.sha256", ".DS_Store", "mem-propagation-data.tar.gz", "forbidden.txt"}
LIMIT = 95 * 1024 ** 2
TEXT_SUFFIXES = {".py", ".sh", ".md", ".tex", ".txt", ".json", ".cff"}
_FORBIDDEN_FILE = ROOT / "tools" / "forbidden.txt"
FORBIDDEN = ([p for p in _FORBIDDEN_FILE.read_text().splitlines() if p.strip()]
             if _FORBIDDEN_FILE.exists() else [])


def files():
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_DIRS) and p.name not in SKIP_FILES:
            yield p


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    man = ROOT / "MANIFEST.sha256"
    if args.check:
        bad = 0
        for line in man.read_text().splitlines():
            digest, rel = line.split("  ", 1)
            p = ROOT / rel
            if not p.is_file() or sha256(p) != digest:
                print(f"MISMATCH {rel}")
                bad += 1
        print("all files match" if not bad else f"{bad} files differ")
        sys.exit(1 if bad else 0)

    problems = []
    for need in ("data/tasks_v2.json", "data/tasks_v2real.json", "results_v2_p2", "results_v2real_p2"):
        if not (ROOT / need).exists():
            problems.append(f"{need} missing: unpack the data archive at the release root first")
    lines, total = [], 0
    for p in files():
        rel = p.relative_to(ROOT).as_posix()
        size = p.stat().st_size
        total += size
        if size > LIMIT:
            problems.append(f"{rel}: {size / 1024 ** 2:.0f} MB, over GitHub's file limit")
        if p.suffix in TEXT_SUFFIXES and size < 5 * 1024 ** 2:
            text = p.read_text(errors="ignore")
            for word in FORBIDDEN:
                if word in text:
                    problems.append(f"{rel}: contains '{word}'")
        lines.append(f"{sha256(p)}  {rel}")
    if problems:
        print("\n".join(problems))
        raise SystemExit("manifest not written")
    man.write_text("\n".join(lines) + "\n")
    print(f"wrote {man} ({len(lines)} files, {total / 1024 ** 2:.1f} MB)")


if __name__ == "__main__":
    main()
