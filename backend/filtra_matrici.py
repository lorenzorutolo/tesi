from __future__ import annotations

import csv
import json
import os
import sys

from estrai_matrici import formatta_matrice


def filtra_matrice(matrice: list[list[float]]) -> list[list[float]]:
    """Tiene la prima riga e le successive con l'argmax sulla stessa colonna."""
    riferimento = matrice[0].index(max(matrice[0]))
    return [riga for riga in matrice
            if riga.index(max(riga)) == riferimento]


def filtra(percorso_input: str, percorso_output: str) -> tuple[int, int, int]:
    """Filtra l'intero CSV; ritorna (domande, righe lette, righe tenute)."""
    with open(percorso_input, newline="", encoding="utf-8") as f_in, \
         open(percorso_output, mode="w", newline="", encoding="utf-8") as f_out:
        lettore = csv.DictReader(f_in)
        writer = csv.writer(f_out)
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
    if len(sys.argv) != 2:
        print("Uso: python filtra_matrici.py <csv_matrici>")
        raise SystemExit(1)
    percorso = sys.argv[1]
    base, estensione = os.path.splitext(percorso)
    output = base + "_filtrato" + estensione
    domande, lette, tenute = filtra(percorso, output)
    print(f"Scritte {domande} domande su '{output}': tenute {tenute} "
          f"ripetizioni su {lette} ({lette - tenute} scartate)")


if __name__ == "__main__":
    main()
