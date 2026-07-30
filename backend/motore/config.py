"""Costanti di runtime del motore di benchmark (indipendenti dal dataset)."""

# Seed unico per la riproducibilita': fissa l'ordine di shuffle dello split
# HuggingFace e i seed passati a Ollama per la generazione delle parafrasi
# (uno per chiamata, derivato da questo). Mettere None per tornare a run non
# deterministiche.
SEED = 42
# Quante parafrasi generare per ogni domanda nei dataset perturbati via
# parafrasi (es. BoolQ): le varianti interrogate sono 1 originale + NUM_PARAFRASI.
# Portato da 10 a 30 dal Test 007 (gemma3:12b) per avere piu' ripetizioni
# sul descrittore; le campagne 001-006 sono state eseguite con 10.
NUM_PARAFRASI = 30

# Endpoint e modello Ollama. MODELLO e' il default, sovrascrivibile a runtime
# con `run.py --modello <tag>`: lo stesso modello risponde alle domande E
# genera le parafrasi (decisione del 2026-07-18), con i suoi parametri di
# sampling di default (temperature/top_p non impostati).
URL_OLLAMA = "http://localhost:11434/api/generate"
MODELLO = "llama3"

# Quanti top token richiedere nei logprobs per stimare la distribuzione
TOP_LOGPROBS = 20

# File CSV di output di default
OUTPUT_CSV = "risultati_benchmark.csv"
