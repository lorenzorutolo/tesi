"""Costanti di runtime del motore di benchmark (indipendenti dal dataset)."""

# Quante domande del dataset processare
NUM_TEST = 3000
# Seed unico per la riproducibilita': fissa sia l'ordine di shuffle dello split
# HuggingFace sia il modulo ``random`` (shuffle delle opzioni nelle varianti).
# Vale per tutti i dataset. Mettere None per tornare a run non deterministiche.
SEED = 42
# Quante varianti (originale + perturbazioni) generare per ogni domanda
RIPETIZIONI_PER_DOMANDA = 5
# Tentativi massimi di perturbazione (parafrasi/shuffle) prima di accettare una
# variante non convergente, cioe' una che cambia la risposta di riferimento.
MAX_TENTATIVI_VARIANTE = 10

# Endpoint e modello Ollama
URL_OLLAMA = "http://localhost:11434/api/generate"
MODELLO = "llama3"  # aggiungere piu modelli possibili

# Quanti top token richiedere nei logprobs per stimare la distribuzione
TOP_LOGPROBS = 20

# File CSV di output di default
OUTPUT_CSV = "risultati_benchmark.csv"
