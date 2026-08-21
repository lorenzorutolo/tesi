"""Server HTTP che espone il motore all'interfaccia web (demo chatbox).

Un solo endpoint: ``POST /api/interroga``. L'utente scrive una domanda
booleana libera e sceglie quante ripetizioni fare: la prima interrogazione
usa la domanda cosi' com'e', ognuna delle successive ne interroga una
parafrasi generata dal modello stesso (come nel benchmark). Ogni esito porta
la sua distribuzione true/false/altro estratta dai logprobs del primo token;
l'aggregazione in ensemble (media delle distribuzioni) e' compito del
frontend. Nessun dataset e nessuna risposta gold: interessa solo che
distribuzione (e quindi che entropia) produce il modello.

Avvio (da dentro la cartella backend/):
    python api.py
Il server resta in ascolto su http://localhost:8000
"""
from __future__ import annotations

from dataclasses import asdict         # converte le dataclass del motore in dizionari serializzabili

from flask import Flask, jsonify, request  # micro-framework HTTP: app, risposta JSON, body richiesta

from motore import Domanda, interroga_e_classifica, parafrasa
from specifiche.boolq import BoolQSpec  # riusata per classi e mappatura token->classe

# Quante ripetizioni al massimo per una richiesta live: il benchmark completo
# e' compito della CLI; l'API e' una demo interattiva e deve restare reattiva.
MAX_RIPETIZIONI_LIVE = 30


class DomandaLiberaSpec(BoolQSpec):
    """Spec demo per la chatbox: stesse classi e stessa mappatura token->classe
    di BoolQ, ma prompt senza passage (la domanda arriva dall'utente, non dal
    dataset, quindi non c'e' nessun contesto da leggere)."""

    nome = "domanda-libera"

    def prompt_risposta(self, d: Domanda) -> str:
        return (
            "You are a strict question answering assistant.\n"
            "Your response must be exactly one word: either 'True' or 'False'. "
            "Do not include any explanations, introductory text, or punctuation.\n\n"
            f"Question: {d.testo}\n\n"
            "Answer:"
        )


# Istanza dell'applicazione Flask e spec unica della demo (e' stateless).
app = Flask(__name__)
_spec = DomandaLiberaSpec()


def _fmt_prob(prob: dict) -> str:
    """Distribuzione grezza (non normalizzata) di una singola ripetizione, con
    piena precisione: e' proprio qui che si vede se un token 'True'/'False'
    porta massa raw < 1 (il resto se ne va nei token fuori vocabolario)."""
    return " · ".join(f"{c} {p * 100:.6f}%" for c, p in prob.items())


def _stampa_ensemble(alternative: list[dict]) -> None:
    """Riepilogo aggregato uguale al calcolo del frontend: ogni ripetizione e'
    prima normalizzata per conto suo (massa raw -> distribuzione), poi si fa la
    media aritmetica. Cosi' si vede da dove esce il valore mostrato a schermo:
    la media puo' stare ben sotto il 100% anche se ogni singola ripetizione e'
    quasi certa, se una manciata di parafrasi ribalta la risposta."""
    somma = {c: 0.0 for c in list(_spec.classi) + ["altro"]}
    valide = 0
    for alt in alternative:
        prob = alt["probabilita"]
        raw = sum(prob.values())
        if raw <= 0:
            continue
        for c in somma:
            somma[c] += prob.get(c, 0.0) / raw
        valide += 1
    if valide == 0:
        return
    media = {c: s / valide for c, s in somma.items()}
    dettaglio = " · ".join(f"{c} {p * 100:.6f}%" for c, p in media.items())
    print(f"  ENSEMBLE (media di {valide} ripetizioni normalizzate): {dettaglio}",
          flush=True)


@app.post("/api/interroga")
def interroga():
    """Interroga il modello live sulla domanda scritta dall'utente.

    Body JSON: {"domanda": "<testo>", "ripetizioni": N}
               (``ripetizioni`` opzionale, default 1, limitato a
               MAX_RIPETIZIONI_LIVE; la prima e' l'originale, le altre N-1
               sono parafrasi)
    Risposta : {"domanda": ..., "alternative": [...]} con una Alternativa
               serializzata per ogni interrogazione riuscita.
    """
    # Legge il body JSON; silent=True evita eccezioni se manca/è malformato -> {}.
    corpo = request.get_json(silent=True) or {}

    testo = (corpo.get("domanda") or "").strip()
    if not testo:
        return jsonify({"errore": "missing or empty 'domanda' field"}), 400

    try:
        ripetizioni = int(corpo.get("ripetizioni", 1))
    except (TypeError, ValueError):
        return jsonify({"errore": "'ripetizioni' must be an integer"}), 400
    ripetizioni = max(1, min(ripetizioni, MAX_RIPETIZIONI_LIVE))

    print(f"\n=== Domanda: {testo!r} | {ripetizioni} ripetizioni ===", flush=True)

    originale = Domanda(testo=testo)
    alternative = []
    for i in range(ripetizioni):
        # Come in BoolQSpec.varianti: ogni parafrasi e' generata dall'ORIGINALE
        # (indipendente, non a catena). Qui senza seed: e' una demo live, non
        # una run riproducibile.
        variante = originale if i == 0 else Domanda(testo=parafrasa(testo))
        alt = interroga_e_classifica(_spec, variante, verbose=False)
        etichetta = "originale" if i == 0 else "parafrasi"
        if alt is None:
            # modello non raggiungibile per questa variante: la salto
            print(f"  rip {i + 1:2d} [{etichetta}] nessuna risposta dal modello", flush=True)
            continue
        print(f"  rip {i + 1:2d} [{etichetta}] {variante.testo!r}", flush=True)
        print(f"           -> {alt.risposta_pulita} | {_fmt_prob(alt.probabilita)}", flush=True)
        alternative.append(asdict(alt))

    if not alternative:
        return jsonify({"errore": "no answer from the model (is Ollama running?)"}), 502

    _stampa_ensemble(alternative)
    return jsonify({"domanda": testo, "alternative": alternative})


# Eseguito solo se lancio il file direttamente (python api.py), non se importato.
if __name__ == "__main__":
    # debug=True: ricarica automatico a ogni modifica del codice + stacktrace dettagliati.
    app.run(port=8000, debug=True)
