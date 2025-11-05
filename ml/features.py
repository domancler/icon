from __future__ import annotations
import pandas as pd, numpy as np

from ml.dates import parse_any_to_datetime, parse_input_date

HOME = "H"
DRAW = "D"
AWAY = "A"

def pick_col(df, names):
    for n in names:
        if n in df.columns: return n
        for c in df.columns:
            if c.lower()==n.lower(): return c
    return None

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza il dataset from20to25.csv.
    Colonne attese: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR
    Output: date, home_team, away_team, home_goals, away_goals, label
    """
    d = df.copy()

    # Rinomino le colonne in modo più leggibile
    d = d.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "label"
    })

    # check per sicurezza
    required = ["date", "home_team", "away_team"]
    if not all(col in d.columns for col in required):
        raise ValueError("Mancano colonne essenziali: date, home_team, away_team.")

    d["date"] = parse_any_to_datetime(d["date"])
    d["label"] = d["label"].astype(str).str.upper().str[0]

    # Mantengo solo le colonne utili
    keep = ["date", "home_team", "away_team", "label", "home_goals", "away_goals"]
    d = d[keep].copy()

    d = d.sort_values("date").reset_index(drop=True)
    return d


def feats(t, n:int=5):
    """
    Calcola le statistiche delle ultime n partite presenti in T.
    """
    if not t:
        return dict(points=0, gf=0, ga=0, gd=0, w=0, d=0, l=0)

    last = pd.DataFrame(t, columns=["date", "pts", "gf", "ga", "w", "d", "l"]).sort_values("date").tail(n)
    d = dict(
        points=int(last["pts"].sum()),
        gf=int(last["gf"].sum()),
        ga=int(last["ga"].sum()),
        gd=int(last["gf"].sum() - last["ga"].sum()),
        w=int(last["w"].sum()),
        d=int(last["d"].sum()),
        l=int(last["l"].sum())
    )
    return d

def rolling_form(d: pd.DataFrame, n:int=5) -> pd.DataFrame:
    """
    Aggiorna il dataframe con nuove colonne in cui per ogni riga ci sarà
    riportata la statistice delle ultime n partite prima di essa. (forma delle squadre)
    """
    # mi assicuro che il dataset sia ordinato per data
    d = d.sort_values("date").reset_index(drop=True).copy()
    teams = d["home_team"].unique()
    hist = {t:[] for t in teams}
    cols = [
        # tutti questi campi fanno riferimento alle ultime n partite
        # in riferimento alla squadra di casa
        "home_points", # punti totalizzati
        "home_gf", # gol fatti (gol for)
        "home_ga", # gol subiti (gol against)
        "home_gd", # differenza goal
        "home_w", # numero di vittorie
        "home_d", # numero di pareggi
        "home_l", # numero di sconfitte

        # in riferimento alla squadra fuoricasa
        "away_points", # punti totalizzati
        "away_gf", # gol fatti
        "away_ga", # gol subiti
        "away_gd", # differenza goal
        "away_w", # numero di vittorie
        "away_d", # numero di pareggi
        "away_l" # numero di sconfitte
    ]

    # setto temporamente tutte le colonne nuove a NaN
    for col in cols:
        d[col] = np.nan

    # per ogni riga
    for i,r in d.iterrows():
        # prendo le squadre
        h,a = r["home_team"], r["away_team"]
        # e calcolo le statistiche delle ultime n partite che
        # precedono questa (della riga attualmente presa)
        hf,af=feats(hist[h], n),feats(hist[a], n)
        # e le salvo sul dt
        d.at[i,"home_points"],d.at[i,"home_gf"],d.at[i,"home_ga"],d.at[i,"home_gd"],d.at[i,"home_w"],d.at[i,"home_d"],d.at[i,"home_l"]=hf.values()
        d.at[i,"away_points"],d.at[i,"away_gf"],d.at[i,"away_ga"],d.at[i,"away_gd"],d.at[i,"away_w"],d.at[i,"away_d"],d.at[i,"away_l"]=af.values()

        # sempre un controllo di sicurezza
        if isinstance(r.get("label"),str) and r["label"] in (HOME, DRAW, AWAY):
            # assegno i punti
            hp=3 if r["label"]==HOME else 1 if r["label"]==DRAW else 0
            ap=3 if r["label"]==AWAY else 1 if r["label"]==DRAW else 0
            # recupero i gol
            hg,ag=int(r["home_goals"]),int(r["away_goals"])
            # aggiorno la storia
            hist[h].append((r["date"],hp,hg,ag,int(r["label"]==HOME),int(r["label"]==DRAW),int(r["label"]==AWAY)))
            hist[a].append((r["date"],ap,ag,hg,int(r["label"]==AWAY),int(r["label"]==DRAW),int(r["label"]==HOME)))
    return d

def gd_and_points_last_matches(d: pd.DataFrame, team: str, dt: pd.Timestamp, rolling: bool = False, n_rolling: int = 5):
    hist = d
    if rolling:
        hist = rolling_form(normalize(d), n=n_rolling)

    # filtro righe prima della data target dove il team compare
    dd = hist[(hist["date"] < dt) & ((hist["home_team"] == team) | (hist["away_team"] == team))]
    if dd.empty:
        return 0, 0  # o puoi gestire come errore/None

    last = dd.sort_values("date").iloc[-1]
    if last["home_team"] == team:
        pts = int(last.get("home_points", 0) or 0)
        gd = int(last.get("home_gd", 0) or 0)
    else:
        pts = int(last.get("away_points", 0) or 0)
        gd = int(last.get("away_gd", 0) or 0)
    return pts, gd

# def get_last_row_for_team(d_roll: pd.DataFrame, team: str, dt) -> pd.Series | None:
#     d = d_roll[(d_roll["date"] < dt) & ((d_roll["home_team"] == team) | (d_roll["away_team"] == team))]
#     if d.empty:
#         return None
#     return d.sort_values("date").iloc[-1]

# def read_team_form(d_roll: pd.DataFrame, team: str, dt, desired_prefix: str) -> dict:
#     """
#     Ritorna un dict con le feature rolling per 'team' < dt,
#     mappate sul prefisso desiderato ('home_' o 'away_') per la partita da predire.
#     Copre: points, gf, ga, gd, w, d, l.
#     """
#     row = get_last_row_for_team(d_roll, team, dt)
#     if row is None:
#         # ritorna tutti zeri se non c'è storia
#         keys = ["points","gf","ga","gd","w","d","l"]
#         return {f"{desired_prefix}{k}": 0 for k in keys}
#
#     # se nell'ultima riga la squadra era home/away, prendo i campi corretti
#     src_prefix = "home_" if row["home_team"] == team else "away_"
#     mapping_keys = ["points","gf","ga","gd","w","d","l"]
#     out = {}
#     for k in mapping_keys:
#         out[f"{desired_prefix}{k}"] = int(row.get(f"{src_prefix}{k}", 0) or 0)
#     return out

def build_matrix(d: pd.DataFrame):
    """
    Costruisce la matrice di apprendimento.
    Filtra le colonne pertitenti (home_*, away_*)
    """
    y = d["label"].dropna()
    feat_cols = [
        c for c in d.columns
        if (c.startswith("home_") or c.startswith("away_"))
           and c not in ("home_team","away_team","home_goals","away_goals")
    ]
    x = d.loc[y.index, feat_cols].apply(pd.to_datetime, errors="coerce").fillna(0.0)
    return x, y, feat_cols
