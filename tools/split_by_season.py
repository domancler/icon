# tools/split_by_season.py  — versione "per stagione"
import argparse
from pathlib import Path
import json
import hashlib
import pandas as pd

from ml.dates import format_datetime_series, parse_any_to_datetime

RAW_DEFAULT = "data/raw/from17to25.csv"
TRAIN_OUT   = "data/processed/train_17-24.csv"
TEST_OUT    = "data/processed/test_24-25.csv"

TRAIN_SEASONS = ["17/18", "18/19", "19/20", "20/21", "21/22", "22/23", "23/24"]
TEST_SEASON   = "24/25"

def md5_of_file(path: str) -> str:
    """
    Calcola l'MD5 di un file in modo da, successivamente,
    verificare che il file non sia cambiato.
    """
    h = hashlib.md5()
    with open(path, "rb") as f: #rb - read binary
        for chunk in iter(lambda: f.read(1 << 16), b""): #b " " - end of file
            h.update(chunk) # type: ignore[arg-type]
    return h.hexdigest()

def build_manifest(src_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Crea il manifest con le informazioni che possono
    tornare utili sul dataset da analizzare.
    """
    m = {
        "source": src_path,
        "source_md5": md5_of_file(src_path),
        "train_seasons": TRAIN_SEASONS,
        "test_season": TEST_SEASON,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
    # Se ci sono le date, aggiungi i range (non obbligatorio)
    for name, df in [("train", train_df), ("test", test_df)]:
        if "Date" in df.columns:
            dd = parse_any_to_datetime(df["Date"])
            if dd.notna().any():
                m[f"{name}_date_min"] = str(dd.min().date())
                m[f"{name}_date_max"] = str(dd.max().date())
    return m

def run(src: str, train_out: str, test_out: str):
    """
    Divide il dataset completo in due parti:
    - train: tutte le stagioni destinate all'addestamento
    - test: tutte le stagioni destinate alla valutazione
    """

    #creo un dataframe dal dataset
    df = pd.read_csv(src)
    if "Season" not in df.columns:
        raise ValueError("Manca la colonna 'Season' nel CSV sorgente.")

    # prendo le righe per il train
    train = df[df["Season"].isin(TRAIN_SEASONS)].copy()
    # e quelle per il test
    test  = df[df["Season"] == TEST_SEASON].copy()

    # per sicurezza
    # train["Date"] = format_datetime_series(parse_any_to_datetime(train["Date"]))
    # test["Date"] = format_datetime_series(parse_any_to_datetime(test["Date"]))

    # creo la cartella madre del file se non esiste e salvo i nuovi dataset
    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_out, index=False)
    test.to_csv(test_out, index=False)

    manifest = build_manifest(src, train, test)
    (Path(train_out).parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Train → {train_out}  ({len(train)} righe, stagioni {TRAIN_SEASONS})")
    print(f"Test  → {test_out}   ({len(test)} righe, stagione {TEST_SEASON})")
    print(f"Manifest → {Path(train_out).parent / 'manifest.json'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=RAW_DEFAULT)
    ap.add_argument("--out_train", default=TRAIN_OUT)
    ap.add_argument("--out_test", default=TEST_OUT)
    args = ap.parse_args()
    run(args.src, args.out_train, args.out_test)

if __name__ == "__main__":
    main()
