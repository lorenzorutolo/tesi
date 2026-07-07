"""Motore di benchmark dataset-agnostico.

Espone l'API pubblica usata dai punti di ingresso (es. ``run.py``) e dalle
specifiche dei dataset.
"""
from .benchmark import (
    elabora_domanda,
    elabora_domanda_esaustiva,
    esegui_benchmark,
    interroga_e_classifica,
    scrivi_riga,
)
from .estrazione import classifica, estrai_distribuzione
from .ollama import interroga_ollama, parafrasa
from .tipi import Alternativa, DatasetSpec, Domanda, RigaBenchmark

__all__ = [
    "esegui_benchmark",
    "elabora_domanda",
    "elabora_domanda_esaustiva",
    "interroga_e_classifica",
    "scrivi_riga",
    "interroga_ollama",
    "parafrasa",
    "estrai_distribuzione",
    "classifica",
    "Domanda",
    "Alternativa",
    "RigaBenchmark",
    "DatasetSpec",
]
