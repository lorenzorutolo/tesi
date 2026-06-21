"""Costanti di runtime del motore di benchmark (indipendenti dal dataset)."""

# Quante domande del dataset processare
NUM_TEST = 3
# Quante varianti (originale + parafrasi) generare per ogni domanda
RIPETIZIONI_PER_DOMANDA = 5
# Tentativi massimi di parafrasi prima di accettare una variante non convergente
MAX_TENTATIVI_PARAFRASI = 10

# Endpoint e modello Ollama
URL_OLLAMA = "http://localhost:11434/api/generate"
MODELLO = "llama3"  # aggiungere piu modelli possibili

# Quanti top token richiedere nei logprobs per stimare la distribuzione
TOP_LOGPROBS = 20

# File CSV di output di default
OUTPUT_CSV = "risultati_benchmark.csv"
