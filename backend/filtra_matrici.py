"""Filtro sul CSV "matrici": tiene solo le ripetizioni coerenti con l'originale.

Legge un ``matrici_*.csv`` prodotto da ``estrai_matrici.py`` e riscrive la
matrice di ogni domanda tenendo SOLO le righe (ripetizioni) il cui argmax cade
sulla stessa colonna dell'argmax della PRIMA riga (la domanda originale):
le parafrasi su cui il modello cambia risposta vengono scartate 

La prima riga e' sempre tenuta (definisce lei il riferimento). ``id`` e
``colonna_corretta`` restano invariati; il formato e' lo stesso a 3 colonne,
quindi il file filtrato si rilegge con lo stesso codice di analisi. Le matrici
possono avere numero di righe diverso tra domanda e domanda.

Uso:
    python backend/filtra_matrici.py matrici_boolq_par.csv
    python backend/filtra_matrici.py matrici_boolq_par.csv -o mio_output.csv
    python backend/filtra_matrici.py matrici_boolq_par_excel.csv --excel

Senza ``-o`` l'output aggiunge il suffisso ``_coerenti`` prima dell'estensione
(``matrici_boolq_par.csv`` -> ``matrici_boolq_par_coerenti.csv``). ``--excel``
usa ``;`` come separatore di colonna in lettura E scrittura (variante per
Excel italiano, vedi estrai_matrici.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from estrai_matrici import formatta_matrice


def nome_output(percorso_input: str) -> str:
    """``matrici_x.csv`` -> ``matrici_x_coerenti.csv`` (stessa cartella)."""
    base, estensione = os.path.splitext(percorso_input)
    return base + "_coerenti" + estensione


def filtra_matrice(matrice: list[list[float]]) -> list[list[float]]:
    """Tiene la prima riga e le successive con l'argmax sulla stessa colonna."""
    riferimento = matrice[0].index(max(matrice[0]))
    return [riga for riga in matrice
            if riga.index(max(riga)) == riferimento]


def filtra(percorso_input: str, percorso_output: str,
           excel: bool = False) -> tuple[int, int, int]:
    """Filtra l'intero CSV; ritorna (domande, righe lette, righe tenute)."""
    delimitatore = ";" if excel else ","
    with open(percorso_input, newline="", encoding="utf-8") as f_in, \
         open(percorso_output, mode="w", newline="", encoding="utf-8") as f_out:
        lettore = csv.DictReader(f_in, delimiter=delimitatore)
        writer = csv.writer(f_out, delimiter=delimitatore)
        writer.writerow(["id", "matrice", "colonna_corretta"])
        domande = lette = tenute = 0
        for riga in lettore:
            matrice = json.loads(riga["matrice"])
            filtrata = filtra_matrice(matrice)
            writer.writerow([riga["id"], formatta_matrice(filtrata),
                             riga["colonna_corretta"]])
            domande += 1
            lette += len(matrice)
            tenute += len(filtrata)
    return domande, lette, tenute


def main() -> None:
    parser = argparse.ArgumentParser(description=(
        "Filtra un CSV matrici tenendo, per ogni domanda, solo le ripetizioni "
        "il cui argmax coincide con quello della prima riga (l'originale)."))
    parser.add_argument("input", help="CSV matrici (matrici_*.csv)")
    parser.add_argument("-o", "--output", default=None,
                        help="percorso del CSV di output (default: <input>_coerenti)")
    parser.add_argument("--excel", action="store_true",
                        help="legge e scrive con separatore ';' (Excel italiano)")
    args = parser.parse_args()

    output = args.output if args.output else nome_output(args.input)
    domande, lette, tenute = filtra(args.input, output, excel=args.excel)
    print(f"Scritte {domande} domande su '{output}': tenute {tenute} "
          f"ripetizioni su {lette} ({lette - tenute} scartate)")


if __name__ == "__main__":
    main()
