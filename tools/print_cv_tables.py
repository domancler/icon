import os
import glob
import sys
from joblib import load
import pandas as pd

def prettify_params(params: dict) -> str:
    """
    Rende più leggibili i parametri, rimuovendo prefissi di pipeline
    (es. 'logisticregression__C' -> 'C') e ordinandoli per chiave.
    """
    clean = {k.split("__")[-1]: v for k, v in params.items()}
    return ", ".join(f"{k}={clean[k]}" for k in sorted(clean))

def guess_model_name(gs) -> str:
    """
    Prova a dedurre il nome del modello dall'ultimo step della pipeline.
    Se non disponibile, torna 'model'.
    """
    try:
        est = gs.best_estimator_
        # pipeline.steps[-1] = (nome_step, oggetto)
        if hasattr(est, "steps") and est.steps:
            return est.steps[-1][1].__class__.__name__
        return est.__class__.__name__
    except Exception:
        return "model"

def collect_best_rows(artifacts_dir: str = "artifacts"):
    """
    Carica tutti i GridSearch salvati come artifacts/grid_*.joblib,
    estrae la riga con rank_test_score == 1 e restituisce un DataFrame.
    """
    paths = sorted(glob.glob(os.path.join(artifacts_dir, "grid_*.joblib")))
    if not paths:
        print(f"[ERRORE] Nessun file trovato in '{artifacts_dir}/grid_*.joblib'.")
        print("Assicurati di aver eseguito le GridSearch e salvato i risultati con joblib.dump(...).")
        sys.exit(1)

    rows = []
    for path in paths:
        try:
            gs = load(path)
        except Exception as e:
            print(f"[WARN] Impossibile caricare {path}: {e}")
            continue

        name_from_file = os.path.basename(path).replace("grid_", "").replace(".joblib", "")
        df = pd.DataFrame(gs.cv_results_)
        # Righe con rank 1 (migliore combinazione)
        best_df = df.loc[df["rank_test_score"] == 1]
        if best_df.empty:
            print(f"[WARN] Nessuna riga con rank_test_score == 1 per {name_from_file}")
            continue

        best = best_df.iloc[0]
        model_label = guess_model_name(gs) or name_from_file

        rows.append({
            "Modello": model_label,                         # es. 'RandomForestClassifier'
            "Alias": name_from_file,                         # es. 'rf'
            "Parametri ottimali": prettify_params(best["params"]),
            "Accuracy media (CV)": round(float(best["mean_test_score"]), 4),
            "Dev. std (CV)": round(float(best["std_test_score"]), 4),
        })

    if not rows:
        print("[ERRORE] Nessun risultato valido estratto dai GridSearch.")
        sys.exit(1)

    # Ordina alfabeticamente per Alias (logreg, rf, svm) o per Accuracy desc
    table = pd.DataFrame(rows).sort_values(by=["Modello", "Alias"]).reset_index(drop=True)
    return table

def print_table(df: pd.DataFrame):
    print("\n=== Tabella 1 — Risultati REALI della Grid Search (TimeSeriesSplit) ===\n")
    # Prova a stampare in formato Markdown se tabulate è presente
    try:
        # pandas usa tabulate se installato
        print(df[["Modello", "Parametri ottimali", "Accuracy media (CV)", "Dev. std (CV)"]].to_markdown(index=False))
    except Exception:
        # fallback: stampa 'plain'
        print(df[["Modello", "Parametri ottimali", "Accuracy media (CV)", "Dev. std (CV)"]].to_string(index=False))

    print("\nLegenda: media e deviazione standard calcolate sui fold temporali (TimeSeriesSplit).")
    print("I parametri corrispondono alla combinazione con rank_test_score = 1 per ciascun modello.\n")

if __name__ == "__main__":
    table = collect_best_rows(artifacts_dir="artifacts")
    print_table(table)
