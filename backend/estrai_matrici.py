"""Convertitore: dal CSV completo del benchmark al CSV "matrici" per l'analisi.

Legge un ``risultati_*.csv`` prodotto da ``motore/benchmark.py`` e ne deriva un
CSV compatto a TRE colonne, una riga per domanda (formato richiesto dal
relatore):

- ``id``: id della domanda;
- ``matrice``: la matrice delle distribuzioni come lista annidata JSON su una
  sola cella, ``[[...],[...],...]``: una riga interna per ripetizione
  (originale per prima, poi parafrasi/permutazioni nell'ordine del CSV
  sorgente), una colonna interna per classe con "altro" per ultima (BoolQ:
  ``true,false,altro``; CommonsenseQA: ``A,B,C,D,E,altro``);
- ``colonna_corretta``: indice 0-based della colonna della matrice
  corrispondente alla risposta gold.

Esempio (BoolQ, colonne interne ``true,false,altro``):

    id,matrice,colonna_corretta
    1,"[[1.54692200e-07,9.99995159e-01,4.89459314e-06],[...],...]",1
    2,"[[9.99743308e-01,2.55966282e-04,7.77698812e-07],[...],...]",0

Il CSV sorgente resta la fonte completa (testi delle parafrasi, ordine delle
permutazioni, risposta argmax): questo file e' una vista derivata, rigenerabile
in ogni momento. Tollera i CSV pre-Test-002 (colonne extra lette per nome).

Uso:
    python backend/estrai_matrici.py risultati_boolq_par.csv
    python backend/estrai_matrici.py risultati_boolq_par.csv -o mio_output.csv
    python backend/estrai_matrici.py risultati_boolq_par.csv --excel

Senza ``-o`` l'output sostituisce il prefisso ``risultati_`` con ``matrici_``
(altrimenti antepone ``matrici_`` al nome del file), nella stessa cartella.

``--excel`` produce la variante per Excel con impostazioni italiane (suffisso
``_excel`` nel nome di default): separatore ``;`` tra le colonne, cosi' il
doppio clic apre il file gia' incolonnato (Excel italiano usa ``;`` come
separatore di elenco). I valori DENTRO la cella della matrice restano col
punto decimale e separati da virgola: la cella e' testo, non numeri, quindi
Excel la mostra cosi' com'e' e la stringa resta JSON valido.

Rilettura in analisi (la cella e' JSON valido):

    df = pd.read_csv("matrici_boolq_par.csv")
    M = np.array(json.loads(df.loc[df["id"] == 1, "matrice"].iloc[0]))
    cc = int(df.loc[df["id"] == 1, "colonna_corretta"].iloc[0])
    p_gold = M[:, cc]        # prob. della gold per ripetizione
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

# alternative_json supera facilmente il limite di default dei campi CSV
# (CommonsenseQA: 120 permutazioni per riga sorgente).
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# Cifre significative dei valori nella matrice: abbastanza per KL/entropia
# (errore relativo ~1e-9), abbastanza poche da tenere la cella compatta.
FORMATO_VALORE = "{:.8e}"


def nome_output(percorso_input: str, excel: bool = False) -> str:
    """``risultati_x.csv`` -> ``matrici_x.csv``; altrimenti ``matrici_<nome>``.
    Con ``excel`` aggiunge il suffisso ``_excel`` prima dell'estensione."""
    cartella, nome = os.path.split(percorso_input)
    if nome.startswith("risultati_"):
        nome = "matrici_" + nome[len("risultati_"):]
    else:
        nome = "matrici_" + nome
    if excel:
        base, estensione = os.path.splitext(nome)
        nome = base + "_excel" + estensione
    return os.path.join(cartella, nome)


def estrai_riga(riga: dict) -> tuple[str, list[str], list[list[float]], int]:
    """Deriva (id, classi, matrice, colonna_corretta) da una riga sorgente.

    Solleva ``ValueError`` se le alternative non condividono le stesse classi o
    se la gold non e' tra le classi: meglio fallire subito che produrre una
    matrice con colonne incoerenti."""
    alternative = json.loads(riga["alternative_json"])
    if not alternative:
        raise ValueError(f"domanda {riga['id']}: nessuna alternativa registrata")

    classi = list(alternative[0]["probabilita"].keys())
    reale = riga["reale"]
    if reale not in classi:
        raise ValueError(f"domanda {riga['id']}: gold '{reale}' non tra le classi {classi}")

    matrice = []
    for alt in alternative:
        prob = alt["probabilita"]
        if list(prob.keys()) != classi:
            raise ValueError(
                f"domanda {riga['id']}: classi incoerenti tra alternative "
                f"({list(prob.keys())} vs {classi})")
        matrice.append([prob[c] for c in classi])

    return riga["id"], classi, matrice, classi.index(reale)


def formatta_matrice(matrice: list[list[float]]) -> str:
    """``[[0.2,0.8],[0.4,0.6]]``: JSON valido, valori a 9 cifre significative."""
    righe = ("[" + ",".join(FORMATO_VALORE.format(v) for v in valori) + "]"
             for valori in matrice)
    return "[" + ",".join(righe) + "]"


def converti(percorso_input: str, percorso_output: str,
             excel: bool = False) -> tuple[int, list[str]]:
    """Converte l'intero CSV; ritorna (numero di domande, classi/colonne).

    Con ``excel`` usa ``;`` come separatore tra le colonne (dialetto per Excel
    italiano); la cella della matrice resta identica (vedi docstring modulo)."""
    with open(percorso_input, newline="", encoding="utf-8") as f_in, \
         open(percorso_output, mode="w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out, delimiter=";" if excel else ",")
        writer.writerow(["id", "matrice", "colonna_corretta"])
        classi_file: list[str] = []
        n = 0
        for riga in csv.DictReader(f_in):
            id_domanda, classi, matrice, colonna_corretta = estrai_riga(riga)
            if not classi_file:
                classi_file = classi
            elif classi != classi_file:
                raise ValueError(
                    f"domanda {id_domanda}: classi {classi} diverse da quelle "
                    f"del resto del file {classi_file}")
            writer.writerow([id_domanda, formatta_matrice(matrice),
                             colonna_corretta])
            n += 1
    return n, classi_file


def main() -> None:
    parser = argparse.ArgumentParser(description=(
        "Deriva dal CSV completo del benchmark il CSV compatto delle matrici: "
        "una riga per domanda con id, matrice JSON e colonna della gold."))
    parser.add_argument("input", help="CSV prodotto dal benchmark (risultati_*.csv)")
    parser.add_argument("-o", "--output", default=None,
                        help="percorso del CSV di output (default: matrici_<input>)")
    parser.add_argument("--excel", action="store_true",
                        help="variante per Excel italiano: separatore ';' tra "
                             "le colonne (default nome: matrici_<input>_excel)")
    args = parser.parse_args()

    output = args.output if args.output else nome_output(args.input, args.excel)
    n, classi = converti(args.input, output, excel=args.excel)
    print(f"Scritte {n} domande su '{output}' "
          f"(colonne della matrice: {','.join(classi)})")


if __name__ == "__main__":
    main()
