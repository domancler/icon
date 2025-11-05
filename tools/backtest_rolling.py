# tools/backtest_rolling.py
# Valuta per "stagione calcistica" (1 luglio -> 30 giugno): per ogni stagione S,
# allena su tutto < 1 luglio S e testa su [1 luglio S, 30 giugno S+1].
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.features import normalize, rolling_form, build_matrix

SPACE = {
  "logreg": {
      "logisticregression__C":[1,3,10],
      "logisticregression__max_iter":[300]
  },
  "svm": {
      "svc__C":[0.5,1,3],
      "svc__kernel": ["linear","rbf"]
  },
  "rf": {
      "n_estimators":[400,800],
      "max_depth":[None,20]
  }
}

def models():
    return {
        "logreg": make_pipeline(StandardScaler(with_mean=False), LogisticRegression(solver="lbfgs")),
        "svm": make_pipeline(StandardScaler(with_mean=False), SVC(probability=True)),
        "rf": RandomForestClassifier(random_state=42)
    }

def season_start(year: int):  # 1 agosto
    """Restituisce la data (circa) di inizio stagione"""
    return pd.Timestamp(year=year, month=8, day=1)

# Nota: eventuali partite fuori stagione (es. Coppa Italia a luglio) vengono escluse dal test set
def season_end(year: int):  # 31 maggio
    """Restituisce la data (circa) di fine stagione"""
    return pd.Timestamp(year=year, month=5, day=31)

def run(csv: str, out: str, model_name: str = "rf"):
    outp = Path(out); outp.mkdir(parents=True, exist_ok=True)
    d = rolling_form(normalize(pd.read_csv(csv)), n=5).sort_values("date")
    years = sorted(set(d["date"].dropna().dt.year))

    rows = []
    for y in years:
        # finestra stagione y/y+1
        start = season_start(y)
        end   = season_end(y+1)
        d_test = d[(d["date"]>=start) & (d["date"]<=end)]
        d_train = d[d["date"] < start]
        if d_test.empty or d_train.empty: continue

        Xtr,ytr,_ = build_matrix(d_train)
        Xte,yte,_ = build_matrix(d_test)

        est = models()[model_name]
        gs = GridSearchCV(est, SPACE[model_name], cv=TimeSeriesSplit(n_splits=5), scoring="accuracy", n_jobs=-1)
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_
        best.fit(Xtr, ytr)

        proba = best.predict_proba(Xte)
        pred = best.classes_[np.argmax(proba, axis=1)]
        rows.append({
            "season": f"{y}-{(y+1)%100:02d}",
            "model": model_name,
            "acc": accuracy_score(yte, pred),
            "macro_f1": f1_score(yte, pred, average="macro"),
            "logloss": log_loss(yte, proba, labels=best.classes_),
            "best_params": gs.best_params_
        })

    df = pd.DataFrame(rows)
    df.to_csv(outp/"rolling_season_eval.csv", index=False)
    print("Saved:", outp/"rolling_season_eval.csv")

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--model", default="rf", choices=["logreg","svm","rf"])
    args = ap.parse_args()
    run(args.csv, args.out, args.model)
