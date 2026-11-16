"""Download and normalise the six public tables of the real task sets.

    python code/prepare_real.py          # download.sh calls it on the login node

For each dataset this writes
    data/real/raw/<file>           the file as downloaded (kept, never modified)
    data/real/<name>.csv           id, group column, all other columns with clean names
    data/real/<name>.json          group column, the numeric columns steps may use, source, SHA-256, row count
A raw file that is already present is not downloaded again, so a file placed there by hand (if a
site is unreachable) is used as is. Row and column counts are asserted against the published sizes,
and if data/real/<name>.json already holds a checksum (as in the release), a raw file or CSV with a
different SHA-256 stops the script rather than produce a different table.

Sources (UCI files are tried at the current static address first, then the legacy one):
    iris      UCI #53    Fisher (1936)                                  150 rows
    wine      UCI #109   Forina et al.                                  178 rows
    diabetes  Efron et al. (2004), scikit-learn's raw copy or the authors' file    442 rows
    heart     UCI #45    Cleveland subset (processed.cleveland.data)    303 rows
    credit    UCI #350   Yeh & Lien (2009)                              30,000 rows
    steel     UCI #198   Buscema et al. (Faults.NNA, 27 features + 7 fault indicators)  1,941 rows
"""
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memprop.common import ROOT, log  # noqa: E402

REAL = ROOT / "data" / "real"
RAW = REAL / "raw"
UCI = "https://archive.ics.uci.edu"

SOURCES = {
    "iris": [(f"{UCI}/static/public/53/iris.zip", "iris.data"),
             (f"{UCI}/ml/machine-learning-databases/iris/iris.data", None)],
    "wine": [(f"{UCI}/static/public/109/wine.zip", "wine.data"),
             (f"{UCI}/ml/machine-learning-databases/wine/wine.data", None)],
    "heart": [(f"{UCI}/static/public/45/heart+disease.zip", "processed.cleveland.data"),
              (f"{UCI}/ml/machine-learning-databases/heart-disease/processed.cleveland.data", None)],
    "credit": [(f"{UCI}/static/public/350/default+of+credit+card+clients.zip", "default of credit card clients.xls"),
               (f"{UCI}/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls", None)],
    "steel": [(f"{UCI}/static/public/198/steel+plates+faults.zip", "Faults.NNA"),
              (f"{UCI}/ml/machine-learning-databases/00198/Faults.NNA", None)],
    "diabetes": [("https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt", None)],
}
RAW_NAME = {"iris": "iris.data", "wine": "wine.data", "heart": "processed.cleveland.data",
            "credit": "default of credit card clients.xls", "steel": "Faults.NNA", "diabetes": "diabetes.tab.txt"}
CITE = {
    "iris": "Fisher RA (1936); UCI Machine Learning Repository, dataset 53",
    "wine": "Forina M et al.; UCI Machine Learning Repository, dataset 109",
    "diabetes": "Efron B, Hastie T, Johnstone I, Tibshirani R (2004) Least angle regression, Ann Stat 32:407-499",
    "heart": "Janosi A, Steinbrunn W, Pfisterer M, Detrano R (1988); UCI Machine Learning Repository, dataset 45",
    "credit": "Yeh IC, Lien CH (2009) Expert Syst Appl 36:2473-2480; UCI Machine Learning Repository, dataset 350",
    "steel": "Buscema M, Terzi S, Tastle W (2010); UCI Machine Learning Repository, dataset 198",
}
STEEL_FEATURES = ["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas", "X_Perimeter",
                  "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity",
                  "Length_of_Conveyer", "TypeOfSteel_A300", "TypeOfSteel_A400", "Steel_Plate_Thickness",
                  "Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index", "Edges_X_Index",
                  "Edges_Y_Index", "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index",
                  "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas"]
STEEL_FAULTS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
WINE_FEATURES = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium", "total_phenols", "flavanoids",
                 "nonflavanoid_phenols", "proanthocyanins", "color_intensity", "hue", "od280_od315", "proline"]
HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
              "slope", "ca", "thal", "num"]


def _get(url):
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research download)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:                                  # noqa: BLE001
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def fetch(name):
    """Path of the raw file, downloading it if absent. Returns (path, source url or 'local file')."""
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / RAW_NAME[name]
    src_file = RAW / (RAW_NAME[name] + ".source")
    if dest.exists() and dest.stat().st_size > 0:
        return dest, (src_file.read_text().strip() if src_file.exists() else "local file")
    errors = []
    for url, member in SOURCES[name]:
        try:
            blob = _get(url)
            if member is not None:
                with zipfile.ZipFile(io.BytesIO(blob)) as z:
                    hits = [n for n in z.namelist() if Path(n).name.lower() == member.lower()]
                    if not hits:
                        raise RuntimeError(f"{member} not in {url} (members: {z.namelist()[:20]})")
                    blob = z.read(hits[0])
            dest.write_bytes(blob)
            src_file.write_text(url + (f" :: {member}" if member else ""))
            log(f"  downloaded {dest.name} from {url}")
            return dest, src_file.read_text().strip()
        except Exception as e:                                  # noqa: BLE001
            errors.append(f"{url}: {e}")
    raise RuntimeError(f"{name}: every source failed; put the file by hand at {dest}\n  " + "\n  ".join(errors))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- per dataset

def load_iris(raw):
    df = pd.read_csv(raw, header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
    df = df.dropna(how="all")
    df["species"] = df["species"].str.replace("Iris-", "", regex=False)
    assert df.shape == (150, 5), df.shape
    return df, "species", ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def load_wine(raw):
    df = pd.read_csv(raw, header=None, names=["cultivar"] + WINE_FEATURES)
    assert df.shape == (178, 14), df.shape
    df["cultivar"] = "cultivar_" + df["cultivar"].astype(int).astype(str)
    return df, "cultivar", ["alcohol", "malic_acid", "alcalinity_of_ash", "magnesium", "total_phenols",
                            "flavanoids", "color_intensity", "proline"]


def read_diabetes():
    """(table, source, sha256). scikit-learn ships the original unscaled table; without it, the authors' file."""
    try:
        from sklearn.datasets import load_diabetes as skl
        df = skl(scaled=False, as_frame=True).frame.rename(columns={"target": "y"})
        return df, "scikit-learn load_diabetes(scaled=False)", None
    except Exception:                        # noqa: BLE001
        raw, source = fetch("diabetes")
        df = pd.read_csv(raw, sep="\t")
        df.columns = [c.strip().lower() for c in df.columns]
        return df, source, sha256(raw)


def load_diabetes(df):
    assert df.shape == (442, 11), df.shape
    assert list(df.columns) == ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6", "y"], list(df.columns)
    df = df.copy()
    df["sex"] = "sex_" + df["sex"].astype(float).round().astype(int).astype(str)
    return df, "sex", ["age", "bmi", "bp", "s1", "s3", "s5", "y"]


def load_heart(raw):
    df = pd.read_csv(raw, header=None, names=HEART_COLS, na_values="?")
    assert df.shape == (303, 14), df.shape
    df["sex"] = df["sex"].map({0.0: "female", 1.0: "male"})
    assert df["sex"].notna().all()
    return df, "sex", ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]


def load_credit(raw):
    df = pd.read_excel(raw, header=1)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"default_payment_next_month": "default_next_month"})
    assert df.shape == (30000, 25), df.shape
    df = df.drop(columns=["id"])
    df["sex"] = df["sex"].map({1: "male", 2: "female"})
    assert df["sex"].notna().all()
    return df, "sex", ["limit_bal", "age", "bill_amt1", "bill_amt2", "pay_amt1", "pay_amt2"]


def load_steel(raw):
    df = pd.read_csv(raw, sep=r"\s+", header=None, names=STEEL_FEATURES + STEEL_FAULTS)
    assert df.shape == (1941, 34), df.shape
    ind = df[STEEL_FAULTS].to_numpy()
    assert (ind.sum(axis=1) == 1).all(), "every plate must carry exactly one fault label"
    df["fault"] = np.array(STEEL_FAULTS)[ind.argmax(axis=1)]
    df = df.drop(columns=STEEL_FAULTS)            # the seven fault indicators folded into one label
    return df, "fault", ["Pixels_Areas", "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity",
                         "Maximum_of_Luminosity", "Steel_Plate_Thickness", "Edges_Index", "Luminosity_Index"]


LOADERS = {"iris": load_iris, "wine": load_wine, "diabetes": load_diabetes, "heart": load_heart,
           "credit": load_credit, "steel": load_steel}


def _check_against_record(name, out_json, digest, csv_digest):
    """Stop if a recorded checksum (data/real/<name>.json from the release) differs from the new file."""
    if not out_json.exists():
        return
    old = json.loads(out_json.read_text())
    for key, new in (("raw_sha256", digest), ("csv_sha256", csv_digest)):
        if old.get(key) and new and old[key] != new:
            raise RuntimeError(f"{name}: {key} {new} differs from the recorded {old[key]} -- "
                               f"the source file has changed; the tasks were built from the recorded one")


def prepare(name):
    out_csv, out_json = REAL / f"{name}.csv", REAL / f"{name}.json"
    if out_csv.exists() and out_json.exists():
        log(f"{name}: already prepared ({out_csv})")
        return True
    if name == "diabetes":
        table, source, digest = read_diabetes()
        res = load_diabetes(table)
    else:
        raw, source = fetch(name)
        res = LOADERS[name](raw)
        digest = sha256(raw)
    df, group, numeric = res
    df = df.reset_index(drop=True)
    other = [c for c in df.columns if c not in [group] + numeric]
    df.insert(0, "id", np.arange(1, len(df) + 1))
    df = df[["id", group] + numeric + other]
    for c in numeric:
        assert pd.api.types.is_numeric_dtype(df[c]), (name, c)
    REAL.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_name(out_csv.name + ".part")
    df.to_csv(tmp, index=False)
    tmp.replace(out_csv)
    csv_digest = sha256(out_csv)
    try:
        _check_against_record(name, out_json, digest, csv_digest)
    except RuntimeError:
        out_csv.unlink()
        raise
    back = pd.read_csv(out_csv)
    info = {"name": name, "group_col": group, "numeric": numeric, "n_rows": int(len(back)),
            "n_cols": int(back.shape[1]), "levels": {str(k): int(v) for k, v in back[group].value_counts().items()},
            "source": source, "raw_sha256": digest, "csv_sha256": csv_digest, "citation": CITE[name]}
    out_json.write_text(json.dumps(info, indent=1))
    log(f"{name}: {len(back)} rows x {back.shape[1]} columns, group '{group}' "
        f"{info['levels']} -> {out_csv.resolve()}")
    return True


def main():
    names = sys.argv[1:] or list(LOADERS)
    failed = []
    for n in names:
        try:
            prepare(n)
        except Exception as e:                           # noqa: BLE001
            failed.append(n)
            print(f"!! {n}: {e}")
    if failed:
        raise SystemExit(f"not prepared: {', '.join(failed)}")


if __name__ == "__main__":
    main()
