"""Interfaccia verso Ollama: interrogazione del modello e servizio di parafrasi.

Entrambi sono indipendenti dal dataset e vivono nel motore.
"""
from __future__ import annotations

from typing import Any

import requests

# Modulo (non nomi singoli): MODELLO puo' essere sovrascritto a runtime da
# run.py (--modello) e la lettura deve vedere il valore aggiornato.
from . import config
from .config import TOP_LOGPROBS, URL_OLLAMA


def interroga_ollama(prompt: str, num_predict: int, logprobs: bool = False,
                     seed: int | None = None) -> dict:
    payload: dict[str, Any] = {
        "model": config.MODELLO,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict},
        "raw": False,
    }
    if seed is not None:
        # Rende deterministico il campionamento (a parita' di versione/hardware
        # del server Ollama); usato per la generazione delle parafrasi.
        payload["options"]["seed"] = seed
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = TOP_LOGPROBS
    try:
        response = requests.post(URL_OLLAMA, json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Errore di connessione a Ollama: {e}")
        return {}


def _prompt_parafrasi(domanda: str) -> str:
    return (
        "You are an expert linguistic assistant. Your only task is to paraphrase the given question.\n"
        "Rewrite the question using different words or sentence structure, but keep the exact same logical meaning and intent.\n"
        "Your response must contain ONLY the new paraphrased question. "
        "Do not include any introductory phrases, explanations, or answers.\n\n"
        f"Question: {domanda}\n\n"
        "Paraphrased Question:"
    )


def parafrasa(testo: str, seed: int | None = None) -> str:
    """Restituisce una parafrasi di ``testo`` tramite il modello.

    In caso di errore o risposta vuota ritorna il testo originale, cosi' il
    chiamante puo' sempre proseguire. Questo servizio viene iniettato nelle spec
    (vedi ``DatasetSpec.varianti``); il motore lo avvolge in una versione che
    passa un ``seed`` deterministico per chiamata."""
    resp = interroga_ollama(_prompt_parafrasi(testo), num_predict=100, seed=seed)
    return resp.get("response", "").strip() or testo
