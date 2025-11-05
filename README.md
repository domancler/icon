# Progetto ICon 2024-25 SerieA-DataLab

**SerieA-DataLab** è un sistema basato su conoscenza (*Knowledge-Based System*) sviluppato nell’ambito del corso di **Ingegneria della Conoscenza** presso l’Università degli Studi di Bari “Aldo Moro”.

Il progetto integra **apprendimento supervisionato**, **ricerca euristica** e **ragionamento logico tramite Prolog** per analizzare i dati storici dei campionati di calcio di **Serie A** e **prevedere l’esito delle partite** (1 = vittoria in casa, X = pareggio, 2 = vittoria fuori casa).  
L’obiettivo è fornire un sistema interpretabile e adattabile al contesto sportivo, in grado di generare previsioni, analisi statistiche e schedine ottimizzate in base a vincoli configurabili.

---

## Installazione

```bash
# 1. Clona il repository
git clone https://github.com/domancler/icon.git
cd SerieA-DataLab

# 2. Crea e attiva un ambiente virtuale (facoltativo ma consigliato)
python -m venv venv
source venv/bin/activate   # su Linux/macOS
venv\Scripts\activate    # su Windows

# 3. Installa le dipendenze
pip install -r requirements.txt
```

---

## Utilizzo rapido

```bash
# Separazione del dataset in train/test
python -m tools.split_by_season

# Addestramento e valutazione dei modelli ML
python -m ml.train_eval

# Predizione una o più partite da stdin
python -m tools.predict_ticket --date "20/09/2025"

# Predizione una o più partite da file
python -m tools.predict_ticket --date "20/09/2025" --infile "matches.txt"

# Generazione schedina con rischio controllato da stdin
python -m search.fixture_selector --date "20/09/2025"

# Generazione schedina con rischio controllato da file
python -m search.fixture_selector --date "20/09/2025" --infile "matches.txt"
```

---

## Tecnologie principali
- **Python 3.11+**
- **pandas**, **numpy**, **scikit-learn**, **imbalanced-learn**
- **pgmpy** (Bayesian Network)
- **pyswip** (interfaccia Prolog)
- **matplotlib** (visualizzazione)
