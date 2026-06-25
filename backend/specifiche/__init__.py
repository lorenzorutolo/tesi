"""Registro delle specifiche di dataset disponibili.

Per aggiungere un dataset: crea un modulo con una classe che implementa
``motore.tipi.DatasetSpec`` e registralo qui sotto in ``REGISTRY``.
"""
from .boolq import BoolQSpec
from .commonsenseqa import CommonsenseQASpec

# nome CLI -> factory della spec
REGISTRY = {
    BoolQSpec.nome: BoolQSpec,
    CommonsenseQASpec.nome: CommonsenseQASpec,
}

__all__ = ["REGISTRY", "BoolQSpec", "CommonsenseQASpec"]
