"""Cuore del motore: interrogazione+classificazione, orchestrazione del
benchmark e scrittura incrementale del CSV.

Percorso di esecuzione UNICO per tutti i dataset: per ogni domanda il modello
viene interrogato su TUTTE le varianti fornite dalla spec (``spec.varianti``),
una volta ciascuna, e ogni esito viene registrato. Niente risposta di
riferimento, niente retry, niente scarti: le decisioni su come trattare le
varianti (es. parafrasi che ribaltano la risposta) si prendono a tempo di
analisi del CSV.

Tutto qui dentro e' agnostico rispetto al dataset: la logica dipende solo dalla
``DatasetSpec`` passata in input.
"""
from __future__ import annotations

import csv
import itertools
import json
from dataclasses import asdict
from typing import Callable

from datasets import load_dataset

from .config import OUTPUT_CSV, SEED, TOP_LOGPROBS
from .estrazione import classifica, estrai_distribuzione
from .ollama import interroga_ollama, parafrasa
from .tipi import Alternativa, DatasetSpec, Domanda, RigaBenchmark


# ============================================================
# INTERROGAZIONE + CLASSIFICAZIONE (generica)
# ============================================================
def _fmt_distribuzione(prob: dict[str, float]) -> str:
    """Riassunto a una riga: classi con prob > 0 in ordine decrescente, poi
    "altro" se non nullo (es. ``B 87.42% · A 6.23% · altro 2.10%``)."""
    voci = sorted(
        ((c, p) for c, p in prob.items() if c != "altro" and p > 0),
        key=lambda cp: cp[1],
        reverse=True,
    )
    if prob.get("altro", 0.0) > 0:
        voci.append(("altro", prob["altro"]))
    return " · ".join(f"{c} {p * 100:.8f}%" for c, p in voci)


def _stampa_interrogazione(
    spec: DatasetSpec,
    d: Domanda,
    prob: dict[str, float],
    dettagli: list[tuple[str, float, str]],
) -> None:
    """Stampa l'esito di una singola interrogazione. Path unico per tutti i
    dataset; l'unico pezzo dataset-specifico e' il blocco delle opzioni, che
    compare solo se la spec espone l'hook ``opzioni_mostrate`` (MC)."""
    opzioni_mostrate = getattr(spec, "opzioni_mostrate", None)
    if opzioni_mostrate is not None:
        print("    Opzioni mostrate (lettere canoniche, ordine rimescolato):")
        for lettera, testo in opzioni_mostrate(d):
            print(f"      {lettera}. {testo}")

    if dettagli:
        print(f"    Top token (Top {TOP_LOGPROBS}):")
        for token, p, etichetta in dettagli:
            print(f"      '{token}' → {p * 100:.8f}% → classe {etichetta}")

    print(f"    Distribuzione classi: {_fmt_distribuzione(prob)}")


def interroga_e_classifica(spec: DatasetSpec, d: Domanda,
                           verbose: bool = True) -> Alternativa | None:
    resp = interroga_ollama(spec.prompt_risposta(d), num_predict=10, logprobs=True)
    if not resp:
        return None
    prob, dettagli = estrai_distribuzione(resp, spec, d)
    if verbose:
        _stampa_interrogazione(spec, d, prob, dettagli)
    # Registra l'ordine di presentazione delle opzioni (solo dataset MC):
    # senza, non si saprebbe quale permutazione ha prodotto questa risposta.
    opzioni_mostrate = getattr(spec, "opzioni_mostrate", None)
    ordine = [lettera for lettera, _ in opzioni_mostrate(d)] if opzioni_mostrate else None
    return Alternativa(
        domanda_alt=d.testo,
        risposta_pulita=classifica(prob),
        probabilita=prob,
        ordine=ordine,
    )


# ============================================================
# ELABORAZIONE DI UNA DOMANDA (tutte le varianti, nessuno scarto)
# ============================================================
def _testo_gold(spec: DatasetSpec, domanda: Domanda) -> str:
    """Suffisso ` → "<testo>"` con il contenuto dell'opzione gold, se la spec
    espone le opzioni (MC); stringa vuota altrimenti (es. BoolQ)."""
    opzioni_mostrate = getattr(spec, "opzioni_mostrate", None)
    if opzioni_mostrate is not None:
        for lettera, testo in opzioni_mostrate(domanda):
            if lettera == domanda.reale:
                return f' → "{testo}"'
    return ""


def elabora_domanda(spec: DatasetSpec, id_domanda: int, riga_dataset: dict,
                    parafrasa_srv: Callable[[str], str] = parafrasa,
                    max_varianti: int | None = None) -> RigaBenchmark:
    """Interroga il modello su TUTTE le varianti della domanda
    (``spec.varianti``), una volta ciascuna, e le registra tutte.

    ``parafrasa_srv`` e' il servizio di parafrasi da iniettare nella spec (il
    default e' quello non seedato; ``esegui_benchmark`` passa la versione con
    seed deterministico per chiamata). ``max_varianti`` limita il numero di
    varianti interrogate (usato dall'API live per restare reattiva).
    L'output di ogni variante e' una riga sola (i top-token completi
    renderebbero il log ingestibile)."""
    domanda = spec.leggi_riga(riga_dataset)
    riga = RigaBenchmark(
        id=id_domanda,
        domanda=domanda.testo,
        reale=domanda.reale,
    )

    print(f"  Q: {domanda.testo}")
    print(f"  Gold: {domanda.reale}{_testo_gold(spec, domanda)}")

    varianti = spec.varianti(domanda, parafrasa_srv)
    if max_varianti is not None:
        varianti = itertools.islice(varianti, max_varianti)

    for i, variante in enumerate(varianti, start=1):
        alt = interroga_e_classifica(spec, variante, verbose=False)
        if alt is None:
            print(f"  var {i:3d}: nessuna risposta dal modello, salto")
            continue
        esito = "✓" if alt.risposta_pulita == domanda.reale else "✗"
        # Per i dataset MC identifica la variante l'ordine delle opzioni; per
        # quelli a parafrasi il testo della variante stessa.
        etichetta = ",".join(alt.ordine) if alt.ordine else variante.testo
        print(f"  var {i:3d} [{etichetta}] → {alt.risposta_pulita} {esito}")
        riga.alternative.append(alt)

    return riga


# ============================================================
# I/O CSV (scrittura incrementale)
# ============================================================
CSV_HEADER = ["id", "domanda", "reale", "alternative_json"]


def scrivi_riga(writer, file_handle, riga: RigaBenchmark) -> None:
    alternative_json = json.dumps([asdict(a) for a in riga.alternative])
    writer.writerow([riga.id, riga.domanda, riga.reale, alternative_json])
    file_handle.flush()


# ============================================================
# ORCHESTRAZIONE
# ============================================================
def _servizio_parafrasi_seedato() -> Callable[[str], str]:
    """Avvolge ``parafrasa`` passando a Ollama un seed deterministico diverso
    per ogni chiamata (SEED * 1_000_000 + contatore progressivo): a parita' di
    run l'ordine delle chiamate e' lo stesso, quindi le parafrasi generate sono
    le stesse. Con SEED None restituisce il servizio non seedato."""
    if SEED is None:
        return parafrasa
    contatore = itertools.count()
    return lambda testo: parafrasa(testo, seed=SEED * 1_000_000 + next(contatore))


def esegui_benchmark(spec: DatasetSpec, filename: str = OUTPUT_CSV,
                     num_test: int | None = None) -> None:
    """Esegue il benchmark su ``spec``: TUTTE le domande dello split (salvo
    override di ``num_test``), ciascuna su TUTTE le varianti fornite dalla
    spec, senza retry ne' scarti."""
    # Con SEED fissato la run e' riproducibile end-to-end: stesso ordine di
    # domande (shuffle HF seedato), varianti deterministiche (le permutazioni
    # MC sono enumerate; le parafrasi sono generate con seed Ollama per
    # chiamata) e risposte deterministiche (argmax sui logprobs del primo
    # token, indipendente dalla temperatura). Condizione verificata
    # empiricamente per le parafrasi: server Ollama nello stesso stato, cioe'
    # istanza appena avviata (come sul cluster, un'istanza per job) — la cache
    # dei prompt di richieste precedenti puo' alterare la generazione — oltre
    # alla solita parita' di versione/hardware.
    parafrasa_srv = _servizio_parafrasi_seedato()

    print(f"Scarico o carico il dataset '{spec.hf_id}' dalla cache...")
    dataset = load_dataset(spec.hf_id)
    dati = dataset[spec.split].shuffle(seed=SEED)

    n = num_test if num_test is not None else len(dati)
    n = min(n, len(dati))
    print(f"\nInizio test su {n} domande "
          f"(tutte le varianti per ciascuna, nessuno scarto)...")
    print(f"Dataset: {spec.nome} | Output incrementale su '{filename}'\n")

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
        file.flush()

        for i in range(n):
            print(f"\nElaborazione Domanda {i + 1}/{n}")
            riga = elabora_domanda(spec, i + 1, dati[i], parafrasa_srv)
            scrivi_riga(writer, file, riga)

    print(f"\nSalvataggio completato su '{filename}'!")
