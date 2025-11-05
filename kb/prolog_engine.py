# kb/prolog_engine.py
from __future__ import annotations
from typing import Dict
from pyswip import Prolog

class PrologKB:
    def __init__(self, kb_path: str = "kb/knowledge.pl"):
        self.prolog = Prolog()
        self.prolog.consult(kb_path)

    def _clean_dynamic_facts(self):
        list(self.prolog.query("retractall(points_last5(_,_))"))
        list(self.prolog.query("retractall(gd_last5(_,_))"))
        list(self.prolog.query("retractall(match(_,_,_))"))

    @staticmethod
    def _norm(team: str) -> str:
        return team.lower().replace(" ", "_")

    def _ask_bool(self, goal: str) -> bool:
        return bool(list(self.prolog.query(goal)))

    def assert_match_facts(self, date: str, home: str, away: str,
                           home_pts5: int, away_pts5: int,
                           home_gd5: int,  away_gd5: int,):
        self._clean_dynamic_facts()
        h = self._norm(home); a = self._norm(away)
        self.prolog.assertz(f"match('{date}', {h}, {a})")
        self.prolog.assertz(f"points_last5({h}, {int(home_pts5)})")
        self.prolog.assertz(f"points_last5({a}, {int(away_pts5)})")
        self.prolog.assertz(f"gd_last5({h}, {int(home_gd5)})")
        self.prolog.assertz(f"gd_last5({a}, {int(away_gd5)})")

    def query_likely(self, home, away):
        h, a = self._norm(home), self._norm(away)
        return {
            "likely_home": self._ask_bool(f"likely_home({h},{a})."),
            "likely_draw": self._ask_bool(f"likely_draw({h},{a})."),
            "likely_away": self._ask_bool(f"likely_away({h},{a})."),
        }

    def logical_prior(self, home: str, away: str) -> Dict[str, float]:
        flags = self.query_likely(home, away)
        base = {"H": 1.0, "D": 1.0, "A": 1.0}
        if flags["likely_home"]: base["H"] += 1.0
        if flags["likely_draw"]: base["D"] += 1.0
        if flags["likely_away"]: base["A"] += 1.0
        s = sum(base.values())
        return {k: v/s for k,v in base.items()}

    def explain(self, home, away) -> list[str]:
        """Opzionale: elenca *perché* (regole/condizioni) sono scattate."""
        h, a = self._norm(home), self._norm(away)
        reasons = []
        if self._ask_bool(f"is_derby({h},{a})."): reasons.append("derby ⇒ draw")
        if self._ask_bool(f"strong_form({h})."):  reasons.append("strong_form(home) ⇒ home")
        if self._ask_bool(f"strong_form({a})."):  reasons.append("strong_form(away) ⇒ away")

        return reasons
