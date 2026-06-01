import csv
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any

import requests
from datasets import load_dataset

# ============================================================
# CONFIGURAZIONE
# ============================================================
NUM_TEST = 3
RIPETIZIONI_PER_DOMANDA = 3
URL_OLLAMA = "http://localhost:11434/api/generate"
MODELLO = "llama3"
DATASET_NAME = "google/boolq"
OUTPUT_CSV = "risultati_benchmark.csv"
TOP_LOGPROBS = 10


# ============================================================
# STRUTTURE DATI
# ============================================================
@dataclass
class Alternativa:
    domanda_alt: str
    risposta_pulita: str
    p_true_raw: float
    p_false_raw: float
    p_altri_raw: float


@dataclass
class RigaBenchmark:
    id: int
    domanda: str
    reale: str
    alternative: list[Alternativa] = field(default_factory=list)


# ============================================================
# PROMPT
# ============================================================
def prompt_risposta(testo: str, domanda: str) -> str:
    return (
        "You are a strict reading comprehension assistant. Read the following passage carefully.\n"
        "Your response must be exactly one word: either 'True' or 'False'. "
        "Do not include any explanations, introductory text, or punctuation.\n\n"
        f"Passage:\n{testo}\n\n"
        f"Question: {domanda}\n\n"
        "Answer:"
    )


def prompt_parafrasi(domanda: str) -> str:
    return (
        "You are an expert linguistic assistant. Your only task is to paraphrase the given question.\n"
        "Rewrite the question using different words or sentence structure, but keep the exact same logical meaning and intent.\n"
        "Your response must contain ONLY the new paraphrased question. "
        "Do not include any introductory phrases, explanations, or answers.\n\n"
        f"Question: {domanda}\n\n"
        "Paraphrased Question:"
    )


# ============================================================
# OLLAMA
# ============================================================
def interroga_ollama(prompt: str, num_predict: int, logprobs: bool = False) -> dict:
    payload: dict[str, Any] = {
        "model": MODELLO,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict},
        "raw": False,
    }
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


def estrai_prob_da_logprobs(resp: dict) -> tuple[float, float, float, list[str]]:
    logprobs = resp.get("logprobs", [])
    if not (isinstance(logprobs, list) and logprobs):
        return 0.0, 0.0, 0.0, []

    p_true = p_false = p_altri = 0.0
    token_grezzi: list[str] = []

    for candidato in logprobs[0].get("top_logprobs", []):
        token = candidato.get("token", "")
        testo = token.strip().lower()
        prob = math.exp(candidato.get("logprob", -100))
        token_grezzi.append(f"'{token}': {prob * 100:.8f}%")

        if "true" in testo:
            p_true += prob
        elif "false" in testo:
            p_false += prob
        else:
            p_altri += prob

    return p_true, p_false, p_altri, token_grezzi


def classifica(p_true: float, p_false: float) -> str:
    if p_true > p_false:
        return "true"
    if p_false > p_true:
        return "false"
    return "altro"


# ============================================================
# BENCHMARK
# ============================================================
def elabora_domanda(id_domanda: int, riga_dataset: dict) -> RigaBenchmark:
    testo = riga_dataset["passage"]
    domanda_originale = riga_dataset["question"]
    riga = RigaBenchmark(
        id=id_domanda,
        domanda=domanda_originale,
        reale=str(riga_dataset["answer"]).lower(),
    )

    domanda_corrente = domanda_originale
    for rep in range(RIPETIZIONI_PER_DOMANDA):
        resp = interroga_ollama(prompt_risposta(testo, domanda_corrente), num_predict=10, logprobs=True)
        if not resp:
            continue

        p_true, p_false, p_altri, token_grezzi = estrai_prob_da_logprobs(resp)

        label = "Original Question " if rep == 0 else "Alternative Question"
        print(f"  - {label} ({rep + 1}): {domanda_corrente}")
        if token_grezzi:
            print(f"    Vettore Token (Top {TOP_LOGPROBS}): [{', '.join(token_grezzi)}]")

        riga.alternative.append(Alternativa(
            domanda_alt=domanda_corrente,
            risposta_pulita=classifica(p_true, p_false),
            p_true_raw=p_true,
            p_false_raw=p_false,
            p_altri_raw=p_altri,
        ))

        # Parafrasi per la prossima ripetizione (saltabile sull'ultima)
        if rep < RIPETIZIONI_PER_DOMANDA - 1:
            resp_pert = interroga_ollama(prompt_parafrasi(domanda_corrente), num_predict=100)
            domanda_corrente = resp_pert.get("response", "").strip() or domanda_corrente

    return riga


# ============================================================
# I/O CSV (scrittura incrementale)
# ============================================================
CSV_HEADER = ["id", "domanda", "reale", "alternative_json"]


def scrivi_riga(writer: csv.writer, file_handle, riga: RigaBenchmark) -> None:
    alternative_json = json.dumps([asdict(a) for a in riga.alternative])
    writer.writerow([riga.id, riga.domanda, riga.reale, alternative_json])
    file_handle.flush()


def esegui_benchmark(filename: str = OUTPUT_CSV) -> None:
    print("Scarico o carico il dataset dalla cache...")
    dataset = load_dataset(DATASET_NAME)
    dati_validazione = dataset["validation"].shuffle()

    print(f"\nInizio test su {NUM_TEST} domande (con {RIPETIZIONI_PER_DOMANDA} ripetizioni l'una)...")
    print(f"Output incrementale su '{filename}'\n")

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
        file.flush()

        for i in range(NUM_TEST):
            print(f"\nElaborazione Domanda {i + 1}/{NUM_TEST}")
            riga = elabora_domanda(i + 1, dati_validazione[i])
            scrivi_riga(writer, file, riga)

    print(f"\nSalvataggio completato su '{filename}'!")


if __name__ == "__main__":
    esegui_benchmark()
