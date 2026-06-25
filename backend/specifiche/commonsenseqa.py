"""Specifica del dataset CommonsenseQA (domande a scelta multipla, 5 opzioni).

Contiene tutto cio' che e' specifico di CommonsenseQA: mappatura dei campi
grezzi, prompt a scelta multipla, riconoscimento della lettera scelta e
generazione delle varianti tramite SHUFFLE delle opzioni.

Scelta di design (vedi anche ``motore.tipi.DatasetSpec``):
- la distribuzione e' in chiavi CANONICHE: la lettera si riferisce sempre alla
  posizione ORIGINALE del contenuto, non a quella mostrata dopo lo shuffle. Cosi'
  se il modello sceglie sempre lo stesso contenuto la classe vincente resta
  stabile tra le varianti e le distribuzioni sono direttamente confrontabili
  senza rimappare nulla;
- la traduzione lettera-mostrata -> lettera-canonica avviene in
  ``classe_di_token``, usando l'ordine delle opzioni salvato in ``Domanda.dati``.
"""
from __future__ import annotations

import random
from typing import Callable

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
        # Le lettere nel prompt seguono la POSIZIONE corrente (A, B, C, ...),
        # non la lettera originale dell'opzione.
        righe = [
            f"{chr(ord('A') + i)}. {testo}"
            for i, (_, testo) in enumerate(_opzioni(d))
        ]
        opzioni_txt = "\n".join(righe)
        return (
            "You are a strict multiple-choice assistant. Read the question and the options carefully.\n"
            "Your response must be exactly one capital letter corresponding to the best option "
            "(e.g. 'A'). Do not include any explanations, introductory text, or punctuation.\n\n"
            f"Question: {d.testo}\n\n"
            f"Options:\n{opzioni_txt}\n\n"
            "Answer:"
        )

    def classe_di_token(self, token: str, d: Domanda) -> str | None:
        # Riconosce la lettera scelta nell'ordine mostrato ("A", " A", "(B)",
        # "c." -> A/B/C) e la traduce nella lettera ORIGINALE dell'opzione
        # (chiave canonica, stabile sul contenuto al di la' dello shuffle).
        t = token.strip().strip("().").upper()
        if not t or t[0] not in self.classi:
            return None
        i = ord(t[0]) - ord("A")
        opzioni = _opzioni(d)
        if 0 <= i < len(opzioni):
            return opzioni[i][0]
        return None

    def genera_variante(self, corrente: Domanda,
                        parafrasa: Callable[[str], str]) -> Domanda:
        # Perturbazione = solo shuffle delle opzioni: stem e risposta reale
        # (lettera originale) non cambiano. ``parafrasa`` non viene usato.
        opzioni = _opzioni(corrente)
        nuovo = list(opzioni)
        if len(nuovo) > 1:
            while True:
                random.shuffle(nuovo)
                if nuovo != opzioni:
                    break
        return Domanda(
            testo=corrente.testo,
            contesto=corrente.contesto,
            reale=corrente.reale,
            dati={"opzioni": nuovo},
        )
