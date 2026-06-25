"""Specifica del dataset BoolQ (domande a risposta true/false).

Contiene tutto cio' che e' specifico di BoolQ: mappatura dei campi grezzi,
prompt di risposta, sinonimi true/false e generazione delle varianti (solo
parafrasi, dato che per il true/false non c'e' nulla da rimescolare).
"""
from __future__ import annotations

from typing import Callable

from motore.tipi import Domanda

SINONIMI_TRUE = {"yes", "truth", "correct", "right"}
SINONIMI_FALSE = {"no", "wrong", "incorrect", "negative"}


class BoolQSpec:
    nome = "boolq"
    hf_id = "google/boolq"
    split = "validation"
    classi = ["true", "false"]

    def leggi_riga(self, raw: dict) -> Domanda:
        return Domanda(
            testo=raw["question"],
            contesto=raw["passage"],
            reale=str(raw["answer"]).lower(),
        )

    def prompt_risposta(self, d: Domanda) -> str:
        return (
            "You are a strict reading comprehension assistant. Read the following passage carefully.\n"
            "Your response must be exactly one word: either 'True' or 'False'. "
            "Do not include any explanations, introductory text, or punctuation.\n\n"
            f"Passage:\n{d.contesto}\n\n"
            f"Question: {d.testo}\n\n"
            "Answer:"
        )

    def classe_di_token(self, token: str, d: Domanda) -> str | None:
        # ``d`` ignorato: il true/false non ha rimescolamento.
        t = token.strip().lower()
        if "true" in t or t in SINONIMI_TRUE:
            return "true"
        if "false" in t or t in SINONIMI_FALSE:
            return "false"
        return None

    def genera_variante(self, corrente: Domanda,
                        parafrasa: Callable[[str], str]) -> Domanda:
        # Per il true/false la sola perturbazione e' la parafrasi della domanda:
        # contesto e risposta reale non cambiano.
        return Domanda(
            testo=parafrasa(corrente.testo),
            contesto=corrente.contesto,
            reale=corrente.reale,
        )
