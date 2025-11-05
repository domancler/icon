# tools/predict_ticket.py
import sys, json, argparse
import pandas as pd
import joblib
import os

from ml.dates import parse_input_date, format_datetime
from ml.features import normalize, rolling_form, gd_and_points_last_matches
from bn.bayes_model import train_bn, predict_bn
from bn.discretization import discretize_series, build_bn_dataframe
from kb.prolog_engine import PrologKB

LABEL = {"H": "1", "D": "X", "A": "2"}
INV   = {"1": "H", "X": "D", "2": "A"}

def parse_lines(lines, default_date: str):
    """
    Elabora le righe ricevute.
    Formati accettati per ogni riga (una partita per riga):
      - "Home - Away" (prende la data passata --date)
      - "Home, Away" (prende la data passata --date)
      - "DD/MM/YYYY, Home, Away"
      - "Home - Away => 1|X|2"  (esito forzato dall'utente)
    """
    default_dt = parse_input_date(default_date) if default_date else None
    out = []
    for raw in lines:
        line = raw.strip()
        if not line: continue

        forced = None
        if "=>" in line:
            line, forced = [p.strip() for p in line.split("=>", 1)]
            forced = forced.upper()
            if forced not in ("1", "X", "2"):
                raise ValueError(f"Esito forzato non valido in: {raw!r}")

        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0][:2].isdigit():
            date = parse_input_date(parts[0])
            home, away = parts[1], parts[2]
        else:
            if " - " in line:
                home, away = [p.strip() for p in line.split(" - ", 1)]
            elif "," in line:
                home, away = [p.strip() for p in line.split(",", 1)]
            else:
                raise ValueError(f"Formato riga non riconosciuto: {raw!r}")
            if not default_date:
                raise ValueError("Manca --date: necessaria se le righe non hanno la data.")
            date = default_dt

        out.append({"date": format_datetime(date), "home": home, "away": away, "forced": forced})
    return out

def safe_predict_ml(est, classes, x_row):
    try:
        proba = est.predict_proba(x_row)[0]
        return dict(zip(classes, proba))
    except Exception:
        return None

def fuse3(p_pl, p_bn, p_ml=None, w_pl=0.25, w_bn=0.35):
    keys = ["H", "D", "A"]

    # w_pl_ = (1.0 - max(0.0, min(1.0, w_pl)))
    # w_bn_ = max(0.0, min(1.0, w_bn))

    if p_ml is None:
        s = w_pl + w_bn
        w_pl /= s; w_bn /= s
        # w_ml = 0.0
        out = {k: w_pl * (p_pl.get(k, 0)) + w_bn * (p_bn.get(k, 0)) for k in keys}
    else:
        w_ml = max(0.0, 1.0 - (w_pl + w_bn))
        out = {k: w_pl * (p_pl.get(k, 0)) + w_bn * (p_bn.get(k, 0)) + w_ml * (p_ml.get(k, 0)) for k in keys}

    s = sum(out.values())
    return {k: (v/s if s>0 else 0.0) for k,v in out.items()}

def _last_row_for_team_before(d_roll: pd.DataFrame, team: str, dt) -> pd.Series | None:
    d = d_roll[(d_roll["date"] < dt) & ((d_roll["home_team"] == team) | (d_roll["away_team"] == team))]
    if d.empty:
        return None
    return d.sort_values("date").iloc[-1]

def read_team_form(d_roll: pd.DataFrame, team: str, dt, desired_prefix: str) -> dict:
    """
    Ritorna un dict con le feature rolling per 'team' PRIMA di dt,
    mappate sul prefisso desiderato ('home_' o 'away_') per la partita da predire.
    Copre: points, gf, ga, gd, w, d, l.
    """
    row = _last_row_for_team_before(d_roll, team, dt)
    if row is None:
        # ritorna tutti zeri se non c'è storia
        keys = ["points","gf","ga","gd","w","d","l"]
        return {f"{desired_prefix}{k}": 0 for k in keys}

    # se nell'ultima riga la squadra era home/away, prendo i campi corretti
    src_prefix = "home_" if row["home_team"] == team else "away_"
    mapping_keys = ["points","gf","ga","gd","w","d","l"]
    out = {}
    for k in mapping_keys:
        out[f"{desired_prefix}{k}"] = int(row.get(f"{src_prefix}{k}", 0) or 0)
    return out

def build_x_row_for_match(d_roll: pd.DataFrame, home: str, away: str, dt, feat_cols: list[str]) -> "pd.DataFrame":
    """Crea una singola riga di feature (DataFrame 1xN) allineata a feat_cols."""
    # leggi forma rolling per home/away (mappate sul prefisso coerente con la partita da predire)
    home_feats = read_team_form(d_roll, home, dt, desired_prefix="home_")
    away_feats = read_team_form(d_roll, away, dt, desired_prefix="away_")

    row_dict = {**home_feats, **away_feats}

    x = pd.DataFrame([row_dict])
    # allinea le colonne al modello (riempi assenti con 0)
    for c in feat_cols:
        if c not in x.columns:
            x[c] = 0
    x = x[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return x

def run(
        *,
        date: str,
        lines: list[str],
        model_path: str = "artifacts/best_model.joblib",
        csv_train: str = "data/processed/train_17-24.csv",
        csv_test: str  = "data/processed/test_24-25.csv",
        w: float = 0.35
) -> dict:
    """
    Calcola le probabilità per le righe fornite e ritorna lo stesso 'result' di main(),
    senza usare argparse / stdin.
    """
    # Concateno i csv di train e test (prendo tutto)
    hist_raw = pd.concat([pd.read_csv(csv_train), pd.read_csv(csv_test)], ignore_index=True)
    # normalizzo e aggiungo colonne
    d_roll = rolling_form(normalize(hist_raw), n=5)
    # costruisco le colonne per la rete bayesiana
    bn_df = build_bn_dataframe(d_roll)
    # e addestro il modello
    bn_model = train_bn(bn_df)

    # carico il modello ML
    est, classes, feat_cols = None, None, None
    try:
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict):
            est = bundle.get("estimator", None)
            classes = bundle.get("classes", getattr(est, "classes_", None))
            feat_cols = bundle.get("feat_cols", getattr(est, "feature_names_in_", None))
        else:
            est = bundle
            classes = getattr(est, "classes_", None)
            feat_cols = getattr(est, "feature_names_in_", None)
    except Exception:
        est, classes, feat_cols = None, None, None

    matches = parse_lines(lines, date)

    # Creo un'istanza di prolog collegata alla KB
    abs_path = os.path.abspath("kb/knowledge.pl")
    kb = PrologKB(abs_path)  # "kb/knowledge.pl")

    rows = []
    ticket_prob = 1.0

    for m in matches:
        date, home, away = m["date"], m["home"], m["away"]
        dt = parse_input_date(date)

        # Ultime 5 per facts Prolog
        h_pts5, h_gd5 = gd_and_points_last_matches(d_roll, home, dt)
        a_pts5, a_gd5 = gd_and_points_last_matches(d_roll, away, dt)
        kb.assert_match_facts(date, home, away, h_pts5, a_pts5, h_gd5, a_gd5)
        p_pl = kb.logical_prior(home, away)  # prior logico {'H','D','A'}

        # Evidenza per BN (discretizza come in build_bn_dataframe)
        s = pd.Series
        try:
            ev = {
                "H_form": str(discretize_series(s([h_pts5])).iloc[0]),
                "A_form": str(discretize_series(s([a_pts5])).iloc[0]),
                "H_gd": str(
                    discretize_series(s([h_gd5]), bins=[-100, -1, 1, 10, 100], labels=["neg", "eq", "pos", "big"]).iloc[
                        0]),
                "A_gd": str(
                    discretize_series(s([a_gd5]), bins=[-100, -1, 1, 10, 100], labels=["neg", "eq", "pos", "big"]).iloc[
                        0]),
            }
        except Exception:
            ev = {"H_form": "mid", "A_form": "mid", "H_gd": "eq", "A_gd": "eq"}
        p_bn = predict_bn(bn_model, ev)  # {'H','D','A'}

        p_ml = None
        if est is not None and classes is not None and feat_cols is not None:
            try:
                x_row = build_x_row_for_match(d_roll, home, away, dt, list(feat_cols))
                proba = est.predict_proba(x_row)[0]
                # mappa sull'ordine H/D/A
                p_ml = {lab: 0.0 for lab in ["H", "D", "A"]}
                for lab, p in zip(classes, proba):
                    p_ml[str(lab)] = float(p)
            except Exception:
                p_ml = None

        # Fusione semplice: se ML manca, pesiamo BN con args.w e Prolog con (1-args.w)
        w_pl = (1.0 - max(0.0, min(1.0, w)))
        w_bn = max(0.0, min(1.0, w))
        p_final = fuse3(p_pl, p_bn, p_ml, w_pl=w_pl, w_bn=w_bn)
        key = INV[m["forced"]] if m.get("forced") else max(p_final, key=p_final.get)
        pick_label = LABEL[key]
        p_pick = float(p_final.get(key, 0.0))
        ticket_prob *= (p_pick if p_pick > 0 else 0.0)

        rows.append({
            "date": date, "home": home, "away": away,
            "pick": pick_label,
            "p_H": float(p_final.get("H", 0.0)),
            "p_D": float(p_final.get("D", 0.0)),
            "p_A": float(p_final.get("A", 0.0)),
            "p_pick": p_pick,
            "forced": bool(m.get("forced"))
        })

    # --- RITORNO STRUTTURATO ---
    result = {
        "matches": rows,
        "ticket_prob": ticket_prob,
        "ticket_prob_percent": f"{ticket_prob * 100:.2f}%"
    }

    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Data default DD/MM/YYYY per le righe senza data", required=False)
    ap.add_argument("--model", default="artifacts/best_model.joblib")
    ap.add_argument("--csv_train", default="data/processed/train_17-24.csv")
    ap.add_argument("--csv_test",  default="data/processed/test_24-25.csv")
    ap.add_argument("--w", type=float, default=0.35, help="Peso BN nella fusione (se ML assente, 1-w va a Prolog)")
    ap.add_argument("--infile", help="File txt con le partite; se omesso legge da stdin")
    ap.add_argument("--json", action="store_true", help="Output JSON invece di testo")
    args = ap.parse_args()
    # Input righe
    if args.infile:
        lines = open(args.infile, "r", encoding="utf-8").read().splitlines()
    else:
        print("Formato accettato:")
        print("  Squadra1 - Squadra2")
        print("  Squadra1, Squadra2")
        print("  DD/MM/YYYY, Squadra1, Squadra2")
        print("Finito l'inserimento: vai a capo e premi CTRL+Z (Windows) o CTRL+D (Linux/macOS).\n")
        lines = sys.stdin.read().splitlines()

    if not lines:
        print("Nessuna partita fornita.")
        return

    # Chiamata alla funzione pura
    result = run(
        date=args.date,
        lines=lines,
        model_path=args.model,
        csv_train=args.csv_train,
        csv_test=args.csv_test,
        w=args.w
    )

    # Output
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for r in result["matches"]:
            forced = " (forzato)" if r["forced"] else ""
            print(f"{r['home']} - {r['away']} => {r['pick']}{forced}  "
                  f"(H={r['p_H']:.2%} X={r['p_D']:.2%} A={r['p_A']:.2%})")
        print("\n=== Schedina ===")
        print(f"Probabilità schedina: {result['ticket_prob']:.2%}")

    return result

if __name__ == "__main__":
    main()
