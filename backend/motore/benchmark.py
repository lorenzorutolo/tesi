"""Cuore del motore: interrogazione+classificazione, loop di convergenza,
orchestrazione del benchmark e scrittura incrementale del CSV.

Tutto qui dentro e' agnostico rispetto al dataset: la logica dipende solo dalla
``DatasetSpec`` passata in input.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict

from datasets import load_dataset

from .config import (
    MAX_TENTATIVI_VARIANTE,
    NUM_TEST,
    OUTPUT_CSV,
    RIPETIZIONI_PER_DOMANDA,
    TOP_LOGPROBS,
)
from .estrazione import classifica, estrai_distribuzione
from .ollama import interroga_ollama, parafrasa
from .tipi import Alternativa, DatasetSpec, Domanda, RigaBenchmark


# ============================================================
# INTERROGAZIONE + CLASSIFICAZIONE (generica)
# ============================================================
def interroga_e_classifica(spec: DatasetSpec, d: Domanda) -> Alternativa | None:
    resp = interroga_ollama(spec.prompt_risposta(d), num_predict=10, logprobs=True)
    if not resp:
        return None
    prob, token_grezzi = estrai_distribuzione(resp, spec, d)
    if token_grezzi:
        print(f"    Vettore Token (Top {TOP_LOGPROBS}): [{', '.join(token_grezzi)}]")
    return Alternativa(
        domanda_alt=d.testo,
        risposta_pulita=classifica(prob),
        probabilita=prob,
    )


# ============================================================
# LOOP DI CONVERGENZA 
# ============================================================
def elabora_domanda(spec: DatasetSpec, id_domanda: int, riga_dataset: dict) -> RigaBenchmark:
    domanda = spec.leggi_riga(riga_dataset)
    riga = RigaBenchmark(
        id=id_domanda,
        domanda=domanda.testo,
        reale=domanda.reale,
    )

    # rep 0: domanda originale, fissa la risposta di riferimento
    print(f"  - Original Question (1): {domanda.testo}")
    alt_originale = interroga_e_classifica(spec, domanda)
    if alt_originale is None:
        return riga
    riga.alternative.append(alt_originale)
    # riferimento = classe (canonica) della domanda originale.
    risposta_riferimento = alt_originale.risposta_pulita
    domanda_corrente = domanda

    # rep >= 1: varianti con retry finche' la risposta coincide con risposta_riferimento
    for rep in range(1, RIPETIZIONI_PER_DOMANDA):
        tentativi_falliti: list[Alternativa] = []
        alt_accettata: Alternativa | None = None
        variante_accettata: Domanda | None = None
        ultima_variante: Domanda = domanda_corrente

        for tentativo in range(MAX_TENTATIVI_VARIANTE):
            variante = spec.genera_variante(domanda_corrente, parafrasa)
            ultima_variante = variante

            print(f"  - Alternative Question ({rep + 1}) [tentativo {tentativo + 1}]: {variante.testo}")
            alt = interroga_e_classifica(spec, variante)
            if alt is None:
                continue

            if alt.risposta_pulita == risposta_riferimento:  # la variante converge: esce dal loop
                alt_accettata = alt
                variante_accettata = variante
                break
            print(f"    [scartata: '{alt.risposta_pulita}' != riferimento '{risposta_riferimento}']")
            tentativi_falliti.append(alt)

        if alt_accettata is not None:
            riga.alternative.append(alt_accettata)
            riga.scartate.extend(tentativi_falliti)
            domanda_corrente = variante_accettata
        elif tentativi_falliti:  # raggiunto il numero massimo: tieni l'ultima e segnala non convergente
            ultima = tentativi_falliti[-1]
            ultima.convergente = False
            print(f"  ! Rep {rep + 1} non convergente dopo {MAX_TENTATIVI_VARIANTE} tentativi: tenuta l'ultima")
            riga.alternative.append(ultima)
            riga.scartate.extend(tentativi_falliti[:-1])
            # la domanda corrente resta l'ultima formulazione tentata
            domanda_corrente = ultima_variante

    return riga


# ============================================================
# I/O CSV (scrittura incrementale)
# ============================================================
CSV_HEADER = ["id", "domanda", "reale", "num_scartate", "alternative_json", "scartate_json"]


def scrivi_riga(writer, file_handle, riga: RigaBenchmark) -> None:
    alternative_json = json.dumps([asdict(a) for a in riga.alternative])
    scartate_json = json.dumps([asdict(a) for a in riga.scartate])
    writer.writerow([riga.id, riga.domanda, riga.reale, len(riga.scartate), alternative_json, scartate_json])
    file_handle.flush()


# ============================================================
# ORCHESTRAZIONE
# ============================================================
def esegui_benchmark(spec: DatasetSpec, filename: str = OUTPUT_CSV) -> None:
    print(f"Scarico o carico il dataset '{spec.hf_id}' dalla cache...")
    dataset = load_dataset(spec.hf_id)
    dati = dataset[spec.split].shuffle()

    print(f"\nInizio test su {NUM_TEST} domande (con {RIPETIZIONI_PER_DOMANDA} ripetizioni l'una)...")
    print(f"Dataset: {spec.nome} | Output incrementale su '{filename}'\n")

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
        file.flush()

        for i in range(NUM_TEST):
            print(f"\nElaborazione Domanda {i + 1}/{NUM_TEST}")
            riga = elabora_domanda(spec, i + 1, dati[i])
            scrivi_riga(writer, file, riga)

    print(f"\nSalvataggio completato su '{filename}'!")
