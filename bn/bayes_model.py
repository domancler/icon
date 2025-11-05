# bn/bayes_model.py
from typing import Any

import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from .discretization import build_bn_dataframe

def train_bn(d: pd.DataFrame):
    """Addestra la rete bayesiana"""
    # Costruisci dataframe discreto (stringhe/categorical) per pgmpy
    data = build_bn_dataframe(d).astype(str)
    # Struttura molto semplice: feature -> Result
    model = DiscreteBayesianNetwork([
        ('H_form','Result'),
        ('A_form','Result'),
        ('H_gd','Result'),
        ('A_gd','Result')
    ])
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    return model

def predict_bn(model: Any, evidence: dict) -> dict:
    """Serve per inferenza, stima la probabilità di ciascun esito data una 'evidenza'."""
    # evidence deve essere discreta (stringhe)
    ev = {k: str(v) for k, v in evidence.items()}
    inf = VariableElimination(model)
    q = inf.query(variables=['Result'], evidence=ev, show_progress=False)
    # q.state_names['Result'] dà l’ordine degli stati
    return {state: float(q.values[i]) for i, state in enumerate(q.state_names['Result'])}
