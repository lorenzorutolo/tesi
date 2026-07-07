"""Server HTTP che espone il motore di benchmark all'interfaccia web.

Un solo endpoint: ``POST /api/interroga``. Riusa esattamente la stessa logica
della CLI (``elabora_domanda``): estrae una domanda casuale dal dataset e la
interroga su un sottoinsieme delle sue varianti (tutte accettate, nessuno
scarto), restituendo il risultato come JSON. Il modello llama3 viene quindi
interrogato dal vivo.

Avvio (da dentro la cartella backend/):
    python api.py
Il server resta in ascolto su http://localhost:8000
"""
from __future__ import annotations

import random                          # per estrarre una domanda casuale dal dataset
from dataclasses import asdict         # converte le dataclass del motore in dizionari serializzabili

from datasets import load_dataset      # caricamento dataset da HuggingFace
from flask import Flask, jsonify, request  # micro-framework HTTP: app, risposta JSON, body richiesta

from motore import elabora_domanda     # stessa logica usata dalla CLI: tutte le varianti, nessuno scarto
from specifiche import REGISTRY        # registro {nome_dataset: classe spec} per il design dataset-agnostico

# Quante varianti al massimo interrogare per una richiesta live: il benchmark
# completo (es. 120 permutazioni MC) e' compito della CLI; l'API e' una demo
# interattiva e deve restare reattiva.
MAX_VARIANTI_LIVE = 10

# Istanza dell'applicazione Flask: a questa registriamo gli endpoint con i decoratori.
app = Flask(__name__)

# cache dei dataset HuggingFace gia' caricati: {nome_dataset: split}
# Cosi' non riscarichiamo/ricarichiamo il dataset a ogni click.
_cache_dataset: dict = {}


def _carica_split(spec):
    # Scarica/carica lo split solo la prima volta; le volte successive riusa la cache.
    if spec.nome not in _cache_dataset:
        # spec.hf_id = identificativo HuggingFace, spec.split = es. "validation"/"train"
        _cache_dataset[spec.nome] = load_dataset(spec.hf_id)[spec.split]
    return _cache_dataset[spec.nome]


@app.post("/api/interroga")
def interroga():
    """Estrae una domanda casuale e la elabora live con il modello.

    Body JSON: {"dataset": "boolq"}  (campo opzionale, default "boolq")
    Risposta : la RigaBenchmark serializzata (id, domanda, reale,
               alternative[]).
    """
    # Legge il body JSON; silent=True evita eccezioni se manca/è malformato -> {}.
    corpo = request.get_json(silent=True) or {}
    nome = corpo.get("dataset", "boolq")             # nome dataset richiesto, default "boolq"

    # Cerca nel registro la classe spec corrispondente al nome richiesto.
    spec_cls = REGISTRY.get(nome)
    if spec_cls is None:
        # Dataset non registrato: risponde 400 (Bad Request) con l'elenco di quelli validi.
        return jsonify({
            "errore": f"dataset '{nome}' sconosciuto",
            "disponibili": sorted(REGISTRY),
        }), 400

    spec = spec_cls()                                    # istanzia la spec del dataset
    dati = _carica_split(spec)                           # ottiene lo split (dalla cache se già caricato)
    indice = random.randrange(len(dati))                 # estrazione casuale di una domanda
    # originale + varianti (limitate per reattività), tutte accettate (interroga llama3)
    riga = elabora_domanda(spec, indice, dati[indice], max_varianti=MAX_VARIANTI_LIVE)

    # Serializza il RigaBenchmark in JSON: asdict trasforma ogni Alternativa (dataclass) in dictionary.
    return jsonify({
        "id": riga.id,
        "domanda": riga.domanda,
        "reale": riga.reale,
        "alternative": [asdict(a) for a in riga.alternative],  # originale + varianti
    })


# Eseguito solo se lancio il file direttamente (python api.py), non se importato.
if __name__ == "__main__":
    # debug=True: ricarica automatico a ogni modifica del codice + stacktrace dettagliati.
    app.run(port=8000, debug=True)
