"""Specifica del dataset CommonsenseQA (domande a scelta multipla, 5 opzioni).

Contiene tutto cio' che e' specifico di CommonsenseQA: mappatura dei campi
grezzi, prompt a scelta multipla, riconoscimento della lettera scelta e
varianti = tutte le permutazioni dell'ordine delle opzioni.

Scelta di design (vedi anche ``motore.tipi.DatasetSpec``):
- le lettere sono INCOLLATE al contenuto: ogni opzione porta con se' la propria
  lettera (la sua chiave canonica) anche dopo lo shuffle. Il prompt mostra quindi
  le lettere canoniche in ordine rimescolato; la perturbazione cambia solo
  l'ORDINE delle righe, non il legame lettera<->contenuto;
- di conseguenza la lettera scelta dal modello E' GIA' la chiave canonica: non
  serve alcun rimappaggio in ``classe_di_token``, e le distribuzioni restano in
  chiavi stabili e direttamente confrontabili tra le varianti.
"""
from __future__ import annotations

import itertools
from typing import Callable, Iterator

from motore.tipi import Domanda


def _opzioni(d: Domanda) -> list[tuple[str, str]]:
    """Opzioni di ``d`` nell'ordine di visualizzazione corrente: lista di coppie
    (lettera_originale, testo)."""
    return d.dati.get("opzioni", [])


class CommonsenseQASpec:
    nome = "commonsenseqa"
    hf_id = "tau/commonsense_qa"
    split = "validation"            # il test ha answerKey vuoto
    classi = ["A", "B", "C", "D", "E"]

    def leggi_riga(self, raw: dict) -> Domanda:
        scelte = raw["choices"]
        # ordine originale: identita' (la lettera mostrata == lettera originale)
        opzioni = list(zip(scelte["label"], scelte["text"]))
        return Domanda(
            testo=raw["question"],
            reale=raw["answerKey"],          # lettera originale dell'opzione corretta
            dati={"opzioni": opzioni},
        )

    def prompt_risposta(self, d: Domanda) -> str:
        # Le lettere seguono il CONTENUTO (canoniche), mostrate nell'ordine
        # rimescolato corrente: dopo lo shuffle compaiono non sequenziali. Cosi'
        # la lettera scelta dal modello e' direttamente la chiave canonica.
        righe = [f"{lettera}. {testo}" for lettera, testo in _opzioni(d)]
        opzioni_txt = "\n".join(righe)
        return (
            "You are a strict multiple-choice assistant. Read the question and the options carefully.\n"
            "Your response must be exactly one capital letter corresponding to the best option "
            "(e.g. 'A'). Do not include any explanations, introductory text, or punctuation.\n\n"
            f"Question: {d.testo}\n\n"
            f"Options:\n{opzioni_txt}\n\n"
            "Answer:"
        )

    def opzioni_mostrate(self, d: Domanda) -> list[tuple[str, str]]:
        # (lettera, testo) nell'ordine di visualizzazione corrente. Le lettere
        # sono gia' canoniche (incollate al contenuto), quindi coincidono con le
        # chiavi della distribuzione. Hook OPZIONALE usato solo per la stampa a
        # terminale: il motore lo invoca via getattr, BoolQ non lo definisce.
        return list(_opzioni(d))

    def classe_di_token(self, token: str, d: Domanda) -> str | None:
        # Corrispondenza ESATTA con una classe dopo aver tolto spazi e
        # parentesi/punto attorno: "A", " a", "(A)", "A." -> A. Token composti
        # come "AE", "BC", "Bs" NON sono la risposta e confluiscono in "altro".
        # Le lettere sono gia' canoniche (incollate al contenuto): ``d`` ignorato.
        t = token.strip().strip("().").upper()
        if t in self.classi:
            return t
        return None

    def varianti(self, d: Domanda,
                 parafrasa: Callable[[str], str]) -> Iterator[Domanda]:
        # Tutte le K! permutazioni dell'ordine delle opzioni, senza ripetizioni
        # (garantito da itertools.permutations), con l'ordine originale per
        # primo cosi' che alternative[0] resti "l'originale" per l'analisi.
        # ``parafrasa`` non viene usato: la perturbazione e' solo l'ordine.
        originale = tuple(_opzioni(d))
        yield d
        for perm in itertools.permutations(originale):
            if perm == originale:
                continue
            yield Domanda(
                testo=d.testo,
                contesto=d.contesto,
                reale=d.reale,
                dati={"opzioni": list(perm)},
            )
