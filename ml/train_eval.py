import argparse, numpy as np, pandas as pd, joblib
from pathlib import Path
from typing import Dict, Any

from joblib import dump
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

from ml.dates import parse_input_date
from ml.features import normalize, rolling_form, build_matrix

SPACE = {
  "logreg": {
      "logisticregression__C": [0.1,1,3,10],        # Coprono ordini di grandezza diversi, da forte a debole regolarizzazione
      "logisticregression__max_iter":[300]
  },
  "svm": {
      "svc__C": [0.5,1,3],                          # serve per vedere se conviene avere un margine ampio (più regolarizzato) o stretto (più flessibile)
      "svc__kernel": ["linear","rbf"]
  },
  "rf": {
      "n_estimators": [200,400,800],
      "max_depth": [None,10,20],
      "min_samples_split": [2,5]
  }
}

def models():
    # La scalatura aiuta i modelli lineari e a margine, evita la centratura
    return {
        "logreg": make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(solver="lbfgs")
        ), # buon solver generico e stabile
        "svm": make_pipeline(
            StandardScaler(with_mean=False),
            SVC(probability=True) # abilita l'output probabilistico (per fondere BN e logloss)
        ),
        "rf": RandomForestClassifier(random_state=42) # numero fisso = reproducibilità
    }

def _cv_metrics(estimator, x, y, cv):
    """Valuta un modello (estimator) in modo affidabile
    :param x: DataFrame con le feature
    :param y: etichette
    """
    accs,f1s,lls=[],[],[]
    for tr, te in cv.split(x):
        e = estimator
        e.fit(x.iloc[tr], y.iloc[tr]) # addestra sulle righe di train
        proba = e.predict_proba(x.iloc[te]) # probabilità sulle righe di test
        pred = e.classes_[np.argmax(proba, axis=1)] # prende la predizione migliore
        accs.append(accuracy_score(y.iloc[te], pred))
        f1s.append(f1_score(y.iloc[te], pred, average="macro"))
        lls.append(log_loss(y.iloc[te], proba, labels=e.classes_))
    return {
        "acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "lls": float(np.mean(lls)), "lls_std": float(np.std(lls))
    }

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return rolling_form(normalize(df), n=5).sort_values('date')

def find_best_model(xtr, ytr, tscv):
    rows = []; best = None; best_acc = -1
    for name, est in models().items():
        gs = GridSearchCV(est, SPACE[name], cv=tscv, scoring="accuracy", n_jobs=-1)
        gs.fit(xtr, ytr)
        # salva l'oggetto GridSearch per uso successivo
        dump(gs, f"artifacts/grid_{name}.joblib")
        # metriche CV sul TRAIN
        cv_result = _cv_metrics(gs.best_estimator_, xtr, ytr, tscv)
        row = dict(model=name,
                   acc=cv_result["acc"], acc_std=cv_result["acc_std"],
                   f1=cv_result["f1"], f1_std=cv_result["f1_std"],
                   logloss=cv_result["lls"], logloss_std=cv_result["lls_std"],
                   best_params=gs.best_params_)
        rows.append(row)
        if row["acc"] > best_acc:
            best_acc = row["acc"]
            best = {
                "model": name,
                "estimator": gs.best_estimator_,
                "classes": gs.best_estimator_.classes_,
                "features": xtr.columns.tolist()
            }
    return rows, best

def save_test(d_test, est, pd, out):
    xte, yte, _ = build_matrix(d_test)
    proba = est.predict_proba(xte)
    pred = est.classes_[np.argmax(proba, axis=1)]
    test_df = pd.DataFrame({
        "metric": ["Accuracy", "Macro-F1", "LogLoss"],
        "value": [
            accuracy_score(yte, pred),
            f1_score(yte, pred, average="macro"),
            log_loss(yte, proba, labels=est.classes_)
        ]
    })
    test_df.to_csv(out / "results_test.csv", index=False)
    print("Saved:", out / "results_test.csv")

def est_fit_and_save(d_train, pd, out, result_name="results_cv.csv"):
    xtr, ytr, _ = build_matrix(d_train)
    tscv = TimeSeriesSplit(n_splits=5)
    rows, best = find_best_model(xtr, ytr, tscv)

    (pd.DataFrame(rows)
     .sort_values("acc", ascending=False)
     .to_csv(out / result_name, index=False)
     )

    # Fit finale su tutto il TRAIN
    est = best["estimator"]
    est.fit(xtr, ytr)
    joblib.dump(best, out / "best_model.joblib")
    print("Saved:", out / "best_model.joblib", "|", out / result_name)
    return est

def run(csv_path: str | None, outdir: str,
        train_end: str | None = None, test_start: str | None = None,
        train_csv: str | None = None, test_csv: str | None = None):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

    # --- MODE A: train/test in file separati ---
    if train_csv is not None:
        d_train = prepare(pd.read_csv(train_csv))
        d_test = prepare(pd.read_csv(test_csv)) if test_csv else pd.DataFrame(columns=d_train.columns)

        if d_train.empty:
            raise ValueError("Train vuoto: controlla")  # --train_csv

        est = est_fit_and_save(d_train, pd, out, "results_cv_train.csv")

        # Test set (se fornito)
        if not d_test.empty:
            save_test(d_test, est, pd, out)
        return

    # --- MODE B: singolo CSV + split per data (come già usavi) ---
    if not csv_path:
        raise ValueError("Devi specificare --csv oppure --train_csv/--test_csv")

    d = prepare(pd.read_csv(csv_path))

    # B1) Solo CV time-aware su tutto (nessun test esplicito)
    if not train_end and not test_start:
        est_fit_and_save(d, pd, out)
        return

    # B2) Split per data (train_end/test_start)
    # dt_train_end  = pd.to_datetime(train_end) if train_end else d["date"].max()
    # dt_test_start = pd.to_datetime(test_start) if test_start else (dt_train_end + pd.Timedelta(days=1))
    dt_train_end = parse_input_date(train_end) if train_end else d["date"].max()
    dt_test_start = parse_input_date(test_start) if test_start else (dt_train_end + pd.Timedelta(days=1))
    d_train = d[d["date"] <= dt_train_end]
    d_test  = d[d["date"] >= dt_test_start]

    if d_train.empty:
        raise ValueError("Train vuoto: controlla")  # --train_csv

    est = est_fit_and_save(d_train, pd, out, "results_cv_train.csv")

    # Test set (se fornito)
    if not d_test.empty:
        save_test(d_test, est, pd, out)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--train_end", default=None)
    ap.add_argument("--test_start", default=None)
    ap.add_argument("--train_csv", default="data/processed/train_17-24.csv")
    ap.add_argument("--test_csv",  default="data/processed/test_24-25.csv")
    args = ap.parse_args()
    run(args.csv, args.out, args.train_end, args.test_start, args.train_csv, args.test_csv)


