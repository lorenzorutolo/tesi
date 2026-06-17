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
RIPETIZIONI_PER_DOMANDA = 5
MAX_TENTATIVI_PARAFRASI = 10
URL_OLLAMA = "http://localhost:11434/api/generate"
MODELLO = "llama3" #aggiungere piu modelli possibili 
DATASET_NAME = "google/boolq"
OUTPUT_CSV = "risultati_benchmark.csv"
TOP_LOGPROBS = 20

SINONIMI_TRUE = {"yes", "truth", "correct", "right"}
SINONIMI_FALSE = {"no", "wrong", "incorrect", "negative"}


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
    convergente: bool = True


@dataclass
class RigaBenchmark:
    id: int
    domanda: str
    reale: str
    alternative: list[Alternativa] = field(default_factory=list)
    scartate: list[Alternativa] = field(default_factory=list)

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
    logprobs = resp.get("logprobs", []) # estrai array logprobs
    if not (isinstance(logprobs, list) and logprobs):
        return 0.0, 0.0, 0.0, []

    p_true = p_false = p_altri = 0.0
    token_grezzi: list[str] = []

    for candidato in logprobs[0].get("top_logprobs", []): # prendi i top K token dalla posizione 1 siccome abbiamo detto che può rispondere solo true / false, ciò che ci interessa è in prima posizione 
        token = candidato.get("token", "") # estraggo il contenuto del token (es. True)
        testo = token.strip().lower()
        prob = math.exp(candidato.get("logprob", -100)) # esponenziale per estrarre probabilità
        token_grezzi.append(f"'{token}': {prob * 100:.8f}%")

        if "true" in testo or testo in SINONIMI_TRUE:
            p_true += prob
        elif "false" in testo or testo in SINONIMI_FALSE:
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
def interroga_e_classifica(testo: str, domanda: str) -> Alternativa | None:
    resp = interroga_ollama(prompt_risposta(testo, domanda), num_predict=10, logprobs=True)
    if not resp:
        return None
    p_true, p_false, p_altri, token_grezzi = estrai_prob_da_logprobs(resp)
    if token_grezzi:
        print(f"    Vettore Token (Top {TOP_LOGPROBS}): [{', '.join(token_grezzi)}]")
    return Alternativa(
        domanda_alt=domanda,
        risposta_pulita=classifica(p_true, p_false),
        p_true_raw=p_true,
        p_false_raw=p_false,
        p_altri_raw=p_altri,
    )


def elabora_domanda(id_domanda: int, riga_dataset: dict) -> RigaBenchmark:
    testo = riga_dataset["passage"]
    domanda_originale = riga_dataset["question"]
    riga = RigaBenchmark(
        id=id_domanda,
        domanda=domanda_originale,
        reale=str(riga_dataset["answer"]).lower(),
    )

    # rep 0: domanda originale, fissa la risposta di riferimento
    print(f"  - Original Question (1): {domanda_originale}")
    alt_originale = interroga_e_classifica(testo, domanda_originale)
    if alt_originale is None:
        return riga
    riga.alternative.append(alt_originale)
    risposta_riferimento = alt_originale.risposta_pulita #impostiamo la risposta di riferimento come quella alla domanda originale
    domanda_corrente = domanda_originale

    # rep >= 1: parafrasi con retry finché la risposta coincide con risposta_riferimento
    for rep in range(1, RIPETIZIONI_PER_DOMANDA):
        tentativi_falliti: list[Alternativa] = []
        alt_accettata: Alternativa | None = None

        for tentativo in range(MAX_TENTATIVI_PARAFRASI):
            resp_pert = interroga_ollama(prompt_parafrasi(domanda_corrente), num_predict=100) #domanda corrente risulta l'ultima parafrasi accettata non è sempre l'originale
            nuova_domanda = resp_pert.get("response", "").strip() or domanda_corrente

            print(f"  - Alternative Question ({rep + 1}) [tentativo {tentativo + 1}]: {nuova_domanda}")
            alt = interroga_e_classifica(testo, nuova_domanda)
            if alt is None:
                continue

            if alt.risposta_pulita == risposta_riferimento: #se la risposta della parafrasi è uguale esce dal loop 
                alt_accettata = alt
                break
            print(f"    [scartata: '{alt.risposta_pulita}' ≠ riferimento '{risposta_riferimento}']") #altrimenti viene inserito nelle domande scartate
            tentativi_falliti.append(alt)

        if alt_accettata is not None:
            riga.alternative.append(alt_accettata)
            riga.scartate.extend(tentativi_falliti)
            domanda_corrente = alt_accettata.domanda_alt
        elif tentativi_falliti: #se raggiungi in numero massimo di parafrasi viene mantenuta la parafrasi corrente e segnata come "convergente=false" mentre le altre precedenti scartate
            ultima = tentativi_falliti[-1]
            ultima.convergente = False
            print(f"  ! Rep {rep + 1} non convergente dopo {MAX_TENTATIVI_PARAFRASI} tentativi: tenuta l'ultima")
            riga.alternative.append(ultima)
            riga.scartate.extend(tentativi_falliti[:-1])
            domanda_corrente = ultima.domanda_alt

    return riga


# ============================================================
# I/O CSV (scrittura incrementale)
# ============================================================
CSV_HEADER = ["id", "domanda", "reale", "num_scartate", "alternative_json", "scartate_json"]


def scrivi_riga(writer: csv.writer, file_handle, riga: RigaBenchmark) -> None:
    alternative_json = json.dumps([asdict(a) for a in riga.alternative])
    scartate_json = json.dumps([asdict(a) for a in riga.scartate])
    writer.writerow([riga.id, riga.domanda, riga.reale, len(riga.scartate), alternative_json, scartate_json])
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
