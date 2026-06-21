"""Server HTTP che espone il motore di benchmark all'interfaccia web.

Un solo endpoint: ``POST /api/interroga``. Riusa esattamente la stessa logica
della CLI (``elabora_domanda``): estrae una domanda casuale dal dataset e ne
calcola originale + ripetizioni (con i retry di parafrasi), restituendo il
risultato come JSON. Il modello llama3 viene quindi interrogato dal vivo.

Avvio (da dentro la cartella backend/):
    python api.py
Il server resta in ascolto su http://localhost:8000
"""
from __future__ import annotations

import random
from dataclasses import asdict

from datasets import load_dataset
from flask import Flask, jsonify, request

from motore import elabora_domanda
from specifiche import REGISTRY

app = Flask(__name__)

# cache dei dataset HuggingFace gia' caricati: {nome_dataset: split}
# Cosi' non riscarichiamo/ricarichiamo il dataset a ogni click.
_cache_dataset: dict = {}


def _carica_split(spec):
    if spec.nome not in _cache_dataset:
        _cache_dataset[spec.nome] = load_dataset(spec.hf_id)[spec.split]
    return _cache_dataset[spec.nome]


@app.post("/api/interroga")
def interroga():
    """Estrae una domanda casuale e la elabora live con il modello.

    Body JSON: {"dataset": "boolq"}  (campo opzionale, default "boolq")
    Risposta : la RigaBenchmark serializzata (id, domanda, reale,
               alternative[], scartate[]).
    """
    corpo = request.get_json(silent=True) or {}
    nome = corpo.get("dataset", "boolq")

    spec_cls = REGISTRY.get(nome)
    if spec_cls is None:
        return jsonify({
            "errore": f"dataset '{nome}' sconosciuto",
            "disponibili": sorted(REGISTRY),
        }), 400

    spec = spec_cls()
    dati = _carica_split(spec)
    indice = random.randrange(len(dati))                 # estrazione casuale
    riga = elabora_domanda(spec, indice, dati[indice])   # originale + ripetizioni + retry

    return jsonify({
        "id": riga.id,
        "domanda": riga.domanda,
        "reale": riga.reale,
        "alternative": [asdict(a) for a in riga.alternative],
        "scartate": [asdict(a) for a in riga.scartate],
    })


if __name__ == "__main__":
    app.run(port=8000, debug=True)
