# ml/dates.py
from __future__ import annotations
import re
import pandas as pd

DATE_FMT = "%d/%m/%Y"  # standard unico per I/O

_DMY_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")  # "DD/MM/YYYY"

def parse_input_date(s: str) -> pd.Timestamp:
    """
    Parser 'rigido' per input utente/CLI: deve essere DD/MM/YYYY.
    """
    s = str(s).strip()
    if not _DMY_RE.match(s):
        raise ValueError(f"Data non valida (usa DD/MM/YYYY): {s!r}")
    dt = pd.to_datetime(s, format=DATE_FMT, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        raise ValueError(f"Impossibile interpretare la data: {s!r}")
    return dt

def parse_any_to_datetime(col: pd.Series) -> pd.Series:
    """
    Parser 'elastico' per import (raw CSV con formati misti).
    Ritorna dtype datetime64[ns]. Non solleva errori: prova vari formati comuni.
    """
    s = col.astype(str).str.strip()
    parsed = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    mask = parsed.notna()
    if not mask.all():
        # prova DD/MM/YY
        p2 = pd.to_datetime(s[~mask], format="%d/%m/%y", errors="coerce")
        parsed.loc[~mask] = p2; mask = parsed.notna()
    if not mask.all():
        # prova ISO YYYY-MM-DD
        p3 = pd.to_datetime(s[~mask], format="%Y-%m-%d", errors="coerce")
        parsed.loc[~mask] = p3; mask = parsed.notna()
    if not mask.all():
        # fallback generico, dayfirst
        p4 = pd.to_datetime(s[~mask], errors="coerce", dayfirst=True)
        parsed.loc[~mask] = p4
    return parsed

def format_datetime_series(col: pd.Series) -> pd.Series:
    """
    Per export/print: formatta in DD/MM/YYYY.
    """
    return pd.to_datetime(col).dt.strftime(DATE_FMT)

def format_datetime(dt: pd.Timestamp) -> str:
    return pd.to_datetime(dt).strftime(DATE_FMT)
