from typing import Sequence, Optional
import pandas as pd

from ml.features import normalize, rolling_form

#troppo pochi semplificano troppo, troppi complicano la rete.
DEFAULT_BINS: tuple[float, ...] = (-1, 5, 9, 12, 16)
DEFAULT_LABELS: tuple[str, ...] = ("low", "mid", "high", "very_high")

def discretize_series(
    s: pd.Series,
    bins: Optional[Sequence[float]] = None,
    labels: Optional[Sequence[str]] = None,
    clip: bool = False,
) -> pd.Series:
    """
    Trasforma una serie numerica (continua) in categorie discrete
    - bins: estremi degli intervalli (len(labels) deve essere len(bins)-1)
    - labels: etichette dei bin
    - clip: se True, taglia i valori fuori dal range dei bins
    """
    bins = tuple(bins) if bins is not None else DEFAULT_BINS
    labels = tuple(labels) if labels is not None else DEFAULT_LABELS

    x = pd.to_numeric(s, errors="coerce")
    if clip:
        x = x.clip(min(bins), max(bins))

    if len(labels) != len(bins) - 1:
        raise ValueError("Le labels devono essere tante quante bins-1")

    return pd.cut(x, bins=bins, labels=list(labels), include_lowest=True)

def build_bn_dataframe(d: pd.DataFrame) -> pd.DataFrame:
    """
    Prende un dataframe (processato e non) e lo trasforma in un
    dataframe adatto per l'apprendimento di una rete bayesiana.
    """
    # il dt dovrebbe arrivare già normalizzato
    dd = d.copy()
    needed_cols = ["H_form", "A_form", "H_gd", "A_gd", "Result"]
    # se sono già presenti le colonne finali (di questa funzione)
    # allora riporta e basta
    if set(needed_cols).issubset(dd.columns):
        return dd[needed_cols].dropna()

    # se invece non sono presenti, controlla che ci siano
    # le colonne originali ed eventualmente ricalcola
    original_cols = {"home_points", "away_points", "home_gd", "away_gd"}
    if not original_cols.issubset(dd.columns):
        dd = rolling_form(normalize(dd), n=5)

    dd['H_form']=discretize_series(dd['home_points'])
    dd['A_form']=discretize_series(dd['away_points'])
    dd['H_gd']=discretize_series(dd['home_gd'], bins=[-100,-1,1,10,100], labels=["neg","eq","pos","big"])
    dd['A_gd']=discretize_series(dd['away_gd'], bins=[-100,-1,1,10,100], labels=["neg","eq","pos","big"])
    dd['Result']=dd['label'].map({'H':'H','D':'D','A':'A'})

    return dd[needed_cols].dropna()
