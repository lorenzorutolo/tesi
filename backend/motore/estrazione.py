"""Estrazione della distribuzione di probabilita' dai logprobs e classificazione.

Entrambe le funzioni sono generiche: non sanno quali siano le classi del
dataset, le chiedono alla ``DatasetSpec``.
"""
from __future__ import annotations

import math

from .config import TOP_LOGPROBS
from .tipi import DatasetSpec, Domanda


def _dist_vuota(spec: DatasetSpec) -> dict[str, float]:
    prob = {c: 0.0 for c in spec.classi}
    prob["altro"] = 0.0
    return prob


def estrai_distribuzione(resp: dict, spec: DatasetSpec, d: Domanda) -> tuple[dict[str, float], list[str]]:
    """Somma le probabilita' dei top token raggruppandole per classe canonica.

    Considera solo la prima posizione (il modello deve rispondere con la classe
    come primo token). Ogni token viene mappato a una classe tramite
    ``spec.classe_di_token``, a cui passiamo la ``Domanda`` corrente ``d`` perche'
    per il multiple-choice la lettera mostrata dipende dallo shuffle; i token non
    riconosciuti confluiscono in "altro".
    """
    prob = _dist_vuota(spec)
    token_grezzi: list[str] = []

    logprobs = resp.get("logprobs", [])
    if not (isinstance(logprobs, list) and logprobs):
        return prob, token_grezzi

    for candidato in logprobs[0].get("top_logprobs", []):
        token = candidato.get("token", "")
        p = math.exp(candidato.get("logprob", -100))
        token_grezzi.append(f"'{token}': {p * 100:.8f}%")

        classe = spec.classe_di_token(token, d)
        prob[classe if classe in prob else "altro"] += p

    return prob, token_grezzi


def classifica(prob: dict[str, float]) -> str:
    """Restituisce la classe vincente (argmax) ignorando "altro".

    In caso di pareggio tra le classi (o nessun segnale) ritorna "altro",
    coerentemente col comportamento storico del benchmark true/false.
    """
    candidati = {c: p for c, p in prob.items() if c != "altro"}
    if not candidati:
        return "altro"

    massimo = max(candidati.values())
    vincenti = [c for c, p in candidati.items() if p == massimo]
    if len(vincenti) != 1 or massimo <= 0:
        return "altro"
    return vincenti[0]


__all__ = ["estrai_distribuzione", "classifica", "TOP_LOGPROBS"]
