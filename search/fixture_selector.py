import argparse
import sys
from tools.predict_ticket import run as predict_ticket

PICK = {"H": 0, "D": 1, "A": 2}

# pick: 0=H, 1=D, 2=A
def _score_tuple(matches, sol, k, min_draw, max_low_margin, low_threshold):
    if len(sol) != k: return -1.0, 0.0, 0.0

    # conteggio pareggi
    draws = sum(1 for _,pick in sol if pick==PICK["D"])

    # LOW sul set di match selezionati (indipendente dal pick)
    lm_cnt = 0
    probs = []
    margin_sum = 0.0
    for i,pick in sol:
        _, ph, pd, pa = matches[i]
        vals = [ph,pd,pa]
        # prob dell’esito migliore:
        probs.append(vals[pick])
        # margin best-second (per "LOW")
        s = sorted(vals, reverse=True)
        margin_sum += (s[0] - s[1])
        if (s[0] - s[1]) < low_threshold:
            lm_cnt += 1

    if draws < min_draw: return -1.0, 0.0, 0.0
    if max_low_margin is not None and lm_cnt > max_low_margin: return -1.0, 0.0, 0.0

    avg_pick = sum(probs) / k
    avg_margin = margin_sum / k

    return avg_pick, avg_margin, draws

def hill_climb(
    matches, k, min_draw, max_low_margin,
    iters=1500, rand_seed=None
):
    import random

    low_threshold = 0.05

    # passandogli un numero fisso, si otterrà sempre la
    # stessa schedina, altrimenti si ottengono schedine diverse
    if rand_seed is not None: random.seed(rand_seed)

    n = len(matches)
    if k>n: raise ValueError("Partite richieste (k) > numero partite fornite")
    if min_draw+max_low_margin>k: raise ValueError("Vincoli (pareggi e rischiose) > numero partite")

    idx = list(range(n))
    # ordino gli indici per prob di pareggio
    idx_sorted_by_pd = sorted(idx, key=lambda i: matches[i][2], reverse=True)
    # e prendo i primi min_draw pareggi
    draws_idx = idx_sorted_by_pd[:min_draw]

    # prendo i rimanenti e li ordino per home e away
    rest = [i for i in idx if i not in draws_idx]
    rest_sorted_by_best_ha = sorted(rest, key=lambda i: max(matches[i][1], matches[i][3]), reverse=True)
    # e prendo i primi k tra pareggi e non
    sel_idx = (draws_idx + rest_sorted_by_best_ha)[:k]

    # costruiamo i pick
    sol = []
    for i in sel_idx:
        _, ph, pd, pa = matches[i]
        if i in draws_idx:
            pick = 1  # forzo X
        else:
            pick = 0 if ph >= pa else 2  # scelgo il migliore tra H/A
        sol.append((i,pick))

    cur_score = _score_tuple(matches, sol, k, min_draw, max_low_margin, low_threshold)

    tries=50
    # ripeti finchè la soluzione non è valida e hai ancora tentativi
    while cur_score[0] < 0 and tries > 0:
        random.shuffle(idx)
        sel_idx = idx[:k]
        sol = []
        # assegna X ai migliori min_draw per pD dentro sel_idx
        topd = sorted(sel_idx, key=lambda i: matches[i][2], reverse=True)[:min_draw]
        for i in sel_idx:
            _, ph, pd, pa = matches[i]
            if i in topd: sol.append((i, 1))
            else: sol.append((i, 0 if ph>=pa else 2))
        cur_score = _score_tuple(matches, sol, k, min_draw, max_low_margin, low_threshold)
        tries -= 1

    # vicinati: (a) flip pick su un match non-X (H<->A), (b) scambia dentro/fuori un match,
    # (c) scambia “ruoli” X/non-X tra due match per mantenere #X = min_draw
    for _ in range(iters):
        move_type = random.choice(["flip","swap","swapX"])
        new = sol[:]

        if move_type=="flip":
            # prendo uno a caso
            j = random.randrange(k)
            i, pick = new[j]
            # se la probabilità maggiore è il pareggio, salto
            if pick==PICK["D"]:
                continue
            # altrimenti aggiungo
            new[j]=(i, PICK["H"] if pick==PICK["A"] else PICK["A"])

        elif move_type=="swap":
            # prendo uno a caso
            out = random.randrange(k)
            out_i, out_pick = new[out]
            # se la probabilità maggiore è il pareggio, salto
            if out_pick==PICK["D"]: continue
            # altrimenti prendo i matches non ancora aggiunti
            pool = [i for i in idx if i not in [ii for ii,_ in new]]
            if not pool: continue
            # prendo uno a caso
            inn = random.choice(pool)
            _, ph, pd, pa = matches[inn]
            new_pick = PICK["H"] if ph>=pa else PICK["A"]
            # e lo aggiungo
            new[out] = (inn, new_pick)

        else:
            # swapX: scambia il “ruolo X” tra due match
            xs = [t for t in new if t[1] == PICK["D"]]
            nxs = [t for t in new if t[1] != PICK["D"]]
            if not xs or not nxs: continue

            jx = random.randrange(len(xs))
            jn = random.randrange(len(nxs))

            ix, _ = xs[jx]
            inon, pnon = nxs[jn]
            # rendo non-X l’attuale X (scegli H/A migliore) e rendo X l’altro
            _, phn, pdn, pan = matches[inon]
            best_non = PICK["H"] if phn>=pan else PICK["A"]
            # ricostruisci 'new'
            new = [(ii,pp) for (ii,pp) in new if ii not in (ix,inon)]
            new.append((ix, PICK["H"] if matches[ix][1]>=matches[ix][3] else PICK["A"]))
            new.append((inon, PICK["D"]))

        sc = _score_tuple(matches, new, k, min_draw, max_low_margin, low_threshold)
        if sc > cur_score:
            sol, cur_score = new, sc

    return sol, cur_score  # sol = [(idx, pick)]

def run(
        date: str,
        lines: list[str],
        num_matches: int = 4,
        min_draw: int = 2,
        max_low_margin: int = 1,
        model_path: str = "artifacts/best_model.joblib",
        csv_train: str = "data/processed/train_17-24.csv",
        csv_test: str = "data/processed/test_24-25.csv",
        w: float = 0.35
):
    res = predict_ticket(
        date=date,
        lines=lines,
        model_path=model_path,
        csv_train=csv_train,
        csv_test=csv_test,
        w=w
    )

    matches = []
    for m in res["matches"]:
        label = f"{m['home']} - {m['away']}"
        matches.append((label, m["p_H"], m["p_D"], m["p_A"]))

    sel_idx, score = hill_climb(matches, k=num_matches, min_draw=min_draw, max_low_margin=max_low_margin)
    print("Schedina con rischio controllato:")
    for (idx, pick) in sel_idx:
        label, pH, pD, pA = matches[idx]
        home, away = label.split(" - ", 1)
        esito = {0: "1", 1: "X", 2: "2"}[pick]
        print(f"{home} - {away} => {esito}  "
              f"(H={pH:.2%} X={pD:.2%} A={pA:.2%})")
    print(f"Score: ({score[PICK["H"]]:.4%}, {score[PICK["D"]]:.4%}, {score[PICK["A"]]})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_matches", type=int, default=4)
    ap.add_argument("--min_draw", type=int, default=2)
    ap.add_argument("--max_low_margin", type=int, default=1)
    ap.add_argument("--date", help="Data default DD/MM/YYYY per le righe senza data", required=False)
    ap.add_argument("--model", default="artifacts/best_model.joblib")
    ap.add_argument("--csv_train", default="data/processed/train_17-24.csv")
    ap.add_argument("--csv_test", default="data/processed/test_24-25.csv")
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

    run(
        date=args.date,
        lines=lines,
        num_matches=args.num_matches,
        min_draw=args.min_draw,
        max_low_margin=args.max_low_margin,
        model_path=args.model,
        csv_train=args.csv_train,
        csv_test=args.csv_test,
        w=args.w
    )

if __name__ == "__main__":
    main()