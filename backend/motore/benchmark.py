"""Cuore del motore: interrogazione+classificazione, loop di convergenza,
orchestrazione del benchmark e scrittura incrementale del CSV.

Tutto qui dentro e' agnostico rispetto al dataset: la logica dipende solo dalla
``DatasetSpec`` passata in input.
"""
from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict

from datasets import load_dataset

from .config import (
    MAX_TENTATIVI_VARIANTE,
    NUM_TEST,
    OUTPUT_CSV,
    RIPETIZIONI_PER_DOMANDA,
    SEED,
    TOP_LOGPROBS,
)
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
# LOOP DI CONVERGENZA 
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


def elabora_domanda(spec: DatasetSpec, id_domanda: int, riga_dataset: dict) -> RigaBenchmark:
    domanda = spec.leggi_riga(riga_dataset)
    riga = RigaBenchmark(
        id=id_domanda,
        domanda=domanda.testo,
        reale=domanda.reale,
    )
    n = RIPETIZIONI_PER_DOMANDA

    print(f"  Q: {domanda.testo}")
    print(f"  Gold: {domanda.reale}{_testo_gold(spec, domanda)}")

    # rep 0: domanda originale, fissa la risposta di riferimento
    print(f"\n  Rip. 1/{n} — originale")
    alt_originale = interroga_e_classifica(spec, domanda)
    if alt_originale is None:
        return riga
    riga.alternative.append(alt_originale)
    # riferimento = classe (canonica) della domanda originale.
    risposta_riferimento = alt_originale.risposta_pulita
    print(f"    Risposta: {risposta_riferimento}   (riferimento fissato)")
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

            print(f"\n  Rip. {rep + 1}/{n} — variante [tentativo {tentativo + 1}]: {variante.testo}")
            alt = interroga_e_classifica(spec, variante)
            if alt is None:
                continue

            if alt.risposta_pulita == risposta_riferimento:  # la variante converge: esce dal loop
                print(f"    Risposta: {alt.risposta_pulita}   ✓ converge con riferimento '{risposta_riferimento}'")
                alt_accettata = alt
                variante_accettata = variante
                break
            print(f"    Risposta: {alt.risposta_pulita}   ✗ scartata (≠ '{risposta_riferimento}')")
            tentativi_falliti.append(alt)

        if alt_accettata is not None:
            riga.alternative.append(alt_accettata)
            riga.scartate.extend(tentativi_falliti)
            domanda_corrente = variante_accettata
        elif tentativi_falliti:  # raggiunto il numero massimo: tieni l'ultima e segnala non convergente
            ultima = tentativi_falliti[-1]
            ultima.convergente = False
            print(f"  ! Rip. {rep + 1}/{n} non convergente dopo {MAX_TENTATIVI_VARIANTE} tentativi: tengo l'ultima (risposta {ultima.risposta_pulita})")
            riga.alternative.append(ultima)
            riga.scartate.extend(tentativi_falliti[:-1])
            # la domanda corrente resta l'ultima formulazione tentata
            domanda_corrente = ultima_variante

    return riga


# ============================================================
# MODALITA' ESAUSTIVA (tutte le varianti, nessun retry/scarto)
# ============================================================
def elabora_domanda_esaustiva(spec: DatasetSpec, id_domanda: int,
                              riga_dataset: dict) -> RigaBenchmark:
    """Interroga il modello su TUTTE le varianti enumerate dalla spec
    (``varianti_esaustive``), una volta ciascuna: niente risposta di
    riferimento, niente retry, niente scarti. L'output di ogni permutazione
    e' una riga sola (i top-token completi renderebbero il log ingestibile)."""
    domanda = spec.leggi_riga(riga_dataset)
    riga = RigaBenchmark(
        id=id_domanda,
        domanda=domanda.testo,
        reale=domanda.reale,
    )

    print(f"  Q: {domanda.testo}")
    print(f"  Gold: {domanda.reale}{_testo_gold(spec, domanda)}")

    for i, variante in enumerate(spec.varianti_esaustive(domanda), start=1):
        alt = interroga_e_classifica(spec, variante, verbose=False)
        if alt is None:
            print(f"  perm {i:3d}: nessuna risposta dal modello, salto")
            continue
        esito = "✓" if alt.risposta_pulita == domanda.reale else "✗"
        ordine = ",".join(alt.ordine) if alt.ordine else "-"
        print(f"  perm {i:3d} [{ordine}] → {alt.risposta_pulita} {esito}")
        riga.alternative.append(alt)

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
def esegui_benchmark(spec: DatasetSpec, filename: str = OUTPUT_CSV,
                     esaustivo: bool = False, num_test: int | None = None) -> None:
    """Esegue il benchmark su ``spec``.

    - modalita' normale: ``num_test`` domande (default NUM_TEST) con
      RIPETIZIONI_PER_DOMANDA ripetizioni e loop di convergenza;
    - modalita' esaustiva (``esaustivo=True``): TUTTE le domande dello split
      (salvo override di ``num_test``) su TUTTE le varianti enumerate dalla
      spec, senza retry ne' scarti. Richiede l'hook ``varianti_esaustive``.
    """
    if esaustivo and getattr(spec, "varianti_esaustive", None) is None:
        raise SystemExit(
            f"Il dataset '{spec.nome}' non supporta la modalita' esaustiva: "
            "la sua spec non definisce 'varianti_esaustive' (lo spazio delle "
            "perturbazioni non e' enumerabile, es. parafrasi)."
        )

    # Con SEED fissato la run e' riproducibile: stesso sottoinsieme/ordine di
    # domande e stesse permutazioni delle opzioni. Anche le risposte sono
    # deterministiche (argmax sui logprobs del primo token, indipendente dalla
    # temperatura): per i dataset senza parafrasi la run e' riproducibile
    # end-to-end. Resta stocastica solo la generazione delle parafrasi (BoolQ),
    # che usa il testo campionato a temperatura default (vedi ollama.parafrasa).
    random.seed(SEED)

    print(f"Scarico o carico il dataset '{spec.hf_id}' dalla cache...")
    dataset = load_dataset(spec.hf_id)
    dati = dataset[spec.split].shuffle(seed=SEED)

    if esaustivo:
        n = num_test if num_test is not None else len(dati)
    else:
        n = num_test if num_test is not None else NUM_TEST
    n = min(n, len(dati))
    if esaustivo:
        print(f"\nInizio test ESAUSTIVO su {n} domande "
              f"(tutte le permutazioni per ciascuna, nessuno scarto)...")
    else:
        print(f"\nInizio test su {n} domande (con {RIPETIZIONI_PER_DOMANDA} ripetizioni l'una)...")
    print(f"Dataset: {spec.nome} | Output incrementale su '{filename}'\n")

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
        file.flush()

        for i in range(n):
            print(f"\nElaborazione Domanda {i + 1}/{n}")
            if esaustivo:
                riga = elabora_domanda_esaustiva(spec, i + 1, dati[i])
            else:
                riga = elabora_domanda(spec, i + 1, dati[i])
            scrivi_riga(writer, file, riga)

    print(f"\nSalvataggio completato su '{filename}'!")
