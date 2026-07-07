"""Punto di ingresso CLI del motore di benchmark.

Percorso unico: per ogni domanda il modello viene interrogato su TUTTE le
varianti fornite dalla spec (le 120 permutazioni delle opzioni per
commonsenseqa; originale + NUM_PARAFRASI parafrasi per boolq), tutte accettate:
niente loop di convergenza ne' scarti (eventuali scarti a tempo di analisi).
Default: tutte le domande dello split.

Uso:
    python backend/run.py [nome_dataset] [file_output.csv] [--num N]

Opzioni:
    --num N         quante domande processare (override; utile per smoke test).

Esempi:
    python backend/run.py                # "boolq", CSV di default
    python backend/run.py boolq risultati_boolq_par.csv
    python backend/run.py commonsenseqa risultati_perm.csv
    python backend/run.py boolq smoke_par.csv --num 2
"""
import sys

# I log usano simboli unicode (→ ✓ ✗): su console Windows cp1252 la print
# esploderebbe, quindi forziamo stdout a UTF-8 (no-op dove e' gia' cosi').
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from motore import esegui_benchmark
from motore.config import OUTPUT_CSV
from specifiche import REGISTRY


def main(argv: list[str]) -> int:
    args = argv[1:]

    num_test = None
    if "--num" in args:
        idx = args.index("--num")
        try:
            num_test = int(args[idx + 1])
        except (IndexError, ValueError):
            print("Uso: --num N (N intero, es. --num 2)")
            return 1
        del args[idx:idx + 2]

    nome = args[0] if len(args) > 0 else "boolq"
    filename = args[1] if len(args) > 1 else OUTPUT_CSV

    spec_cls = REGISTRY.get(nome)
    if spec_cls is None:
        disponibili = ", ".join(sorted(REGISTRY))
        print(f"Dataset '{nome}' sconosciuto. Disponibili: {disponibili}")
        return 1

    esegui_benchmark(spec_cls(), filename, num_test=num_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
