"""Punto di ingresso CLI del motore di benchmark.

Uso:
    python backend/run.py [nome_dataset] [file_output.csv]

Esempi:
    python backend/run.py            # usa "boolq" e il CSV di default
    python backend/run.py boolq
    python backend/run.py boolq risultati.csv
"""
import sys

from motore import esegui_benchmark
from motore.config import OUTPUT_CSV
from specifiche import REGISTRY


def main(argv: list[str]) -> int:
    nome = argv[1] if len(argv) > 1 else "boolq"
    filename = argv[2] if len(argv) > 2 else OUTPUT_CSV

    spec_cls = REGISTRY.get(nome)
    if spec_cls is None:
        disponibili = ", ".join(sorted(REGISTRY))
        print(f"Dataset '{nome}' sconosciuto. Disponibili: {disponibili}")
        return 1

    esegui_benchmark(spec_cls(), filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
