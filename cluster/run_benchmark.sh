#!/usr/bin/env bash
# Orchestratore per Thor: avvia il server Ollama, scarica il modello, esegue il
# benchmark su ENTRAMBI i dataset (boolq parafrasi, poi commonsenseqa
# permutazioni) dentro Apptainer, poi ferma il server.
#
# Uso:
#   bash run_benchmark.sh --<modello>
# Il modello e' OBBLIGATORIO e si sceglie con un flag-alias DESCRITTIVO (nome
# completo del modello, lo stesso che entra nei nomi dei CSV di output);
# stesso modello per risposte e parafrasi. L'alias NON coincide sempre col tag
# Ollama: per Gemma i tag "*-instruct"/"*-it" NON esistono nella libreria
# Ollama — il tag di default e' GIA' la variante instruction-tuned (i base
# hanno suffisso "-text"). Verificato su ollama.com il 2026-07-26.
#   --llama3                -> llama3
#   --llama3.1-8b           -> llama3.1:8b
#   --qwen2.5-14b-instruct  -> qwen2.5:14b-instruct
#   --gemma2-9b-instruct    -> gemma2:9b
#   --gemma3-12b-it         -> gemma3:12b
# Per un tag non in tabella: --nomeModello TAG (es. --nomeModello mistral:7b);
# in quel caso il suffisso CSV e' il tag con ':' e '/' -> '-'.
# Variabili opzionali:
#   BASE=...     cartella di lavoro in scratch (default: scratch reale dell'utente)
#   REPO=...     copia del repo            (default: $BASE/repo)
#   USE_GPU=1    aggiunge --nv (nodi con GPU NVIDIA)
#   SMOKE=1      smoke test: 1 domanda per dataset su CSV separati
set -euo pipefail

# --- Config -------------------------------------------------------------------
# Percorso scratch REALE, non il symlink ~/scratch: la guida Thor avverte che
# il symlink dentro i container causa comportamenti incoerenti. Stesso percorso
# fisico di prima, quindi immagini e cache esistenti restano valide.
BASE="${BASE:-/mnt/beegfs/scratch/$USER/benchmark-thor}"
REPO="${REPO:-$BASE/repo}"
INSTANCE=ollama
MODELLO=""
TAG_FILE=""

uso() {
  echo "Uso: bash run_benchmark.sh --<modello>"
  echo "Modelli disponibili (alias -> tag Ollama):"
  echo "  --llama3                -> llama3"
  echo "  --llama3.1-8b           -> llama3.1:8b"
  echo "  --qwen2.5-14b-instruct  -> qwen2.5:14b-instruct"
  echo "  --gemma2-9b-instruct    -> gemma2:9b"
  echo "  --gemma3-12b-it         -> gemma3:12b"
  echo "Tag arbitrario: --nomeModello TAG (es. --nomeModello mistral:7b)"
}

# Per ogni alias si fissano sia il tag Ollama sia il suffisso dei CSV: per
# Gemma i due differiscono (il tag "9b-instruct"/"12b-it" non esiste, ma nel
# nome file si tiene l'alias descrittivo, coerente col nome del modello).
while [ $# -gt 0 ]; do
  case "$1" in
    --llama3)                MODELLO=llama3;               TAG_FILE=llama3;               shift ;;
    --llama3.1-8b)           MODELLO=llama3.1:8b;          TAG_FILE=llama3.1-8b;          shift ;;
    --qwen2.5-14b-instruct)  MODELLO=qwen2.5:14b-instruct; TAG_FILE=qwen2.5-14b-instruct; shift ;;
    --gemma2-9b-instruct)    MODELLO=gemma2:9b;            TAG_FILE=gemma2-9b-instruct;   shift ;;
    --gemma3-12b-it)         MODELLO=gemma3:12b;           TAG_FILE=gemma3-12b-it;        shift ;;
    --nomeModello)
      [ $# -ge 2 ] || { echo "Uso: --nomeModello TAG (es. --nomeModello llama3.1:8b)"; exit 1; }
      MODELLO="$2"; shift 2 ;;
    *)
      echo "Argomento sconosciuto: $1"; uso; exit 1 ;;
  esac
done

# Modello obbligatorio: meglio fermarsi subito che scoprire dopo giorni di
# aver lanciato la campagna col modello sbagliato.
[ -n "$MODELLO" ] || { echo "Nessun modello indicato."; uso; exit 1; }

# Suffisso per i nomi dei CSV quando il modello arriva da --nomeModello (gli
# alias lo fissano gia'): il tag Ollama contiene ':' (e a volte '/'),
# caratteri scomodi nei nomi di file.
[ -n "$TAG_FILE" ] || TAG_FILE=$(echo "$MODELLO" | tr ':/' '--')

echo ">> Modello: $MODELLO (suffisso CSV: $TAG_FILE)"

NV_FLAG=""
[ "${USE_GPU:-0}" = "1" ] && NV_FLAG="--nv"

mkdir -p "$BASE/images" "$BASE/ollama_models" "$BASE/hf_cache" "$BASE/out"

# --- 1. Avvia il server Ollama come istanza in background ---------------------
echo ">> Avvio del server Ollama..."
apptainer instance start $NV_FLAG \
  --bind "$BASE/ollama_models:/models" \
  "$BASE/images/ollama.sif" "$INSTANCE"

# Ferma sempre l'istanza all'uscita, anche in caso di errore.
cleanup() { apptainer instance stop "$INSTANCE" 2>/dev/null || true; }
trap cleanup EXIT

# Fail-fast: se e' richiesta la GPU ma il nodo non la vede, meglio morire
# subito che scoprire dopo giorni che la run e' andata su CPU (gia' successo).
if [ -n "$NV_FLAG" ]; then
  echo ">> Verifica GPU nell'istanza..."
  apptainer exec "instance://$INSTANCE" nvidia-smi \
    || { echo "ERRORE: USE_GPU=1 ma nessuna GPU visibile nell'istanza."; exit 1; }
fi

# 'apptainer instance start' esegue il %startscript, NON il %runscript.
# ollama.def definisce solo %runscript, quindi 'ollama serve' non parte da solo:
# lo lanciamo a mano dentro l'istanza, in background.
echo ">> Avvio 'ollama serve' dentro l'istanza..."
apptainer exec "instance://$INSTANCE" ollama serve >"$BASE/ollama_serve.log" 2>&1 &

# --- 2. Attendi che il server risponda ---------------------------------------
echo ">> Attendo che Ollama sia pronto..."
until apptainer exec "instance://$INSTANCE" ollama list >/dev/null 2>&1; do
  sleep 1
done

# Con che backend e' partito il server? Deve dire library=cuda, non library=cpu.
echo ">> Backend di inferenza rilevato da Ollama:"
grep -m1 "inference compute" "$BASE/ollama_serve.log" || echo "   (riga 'inference compute' non ancora nel log)"

# --- 3. Scarica il modello (no-op se gia' in cache) --------------------------
echo ">> Scarico il modello $MODELLO..."
apptainer exec "instance://$INSTANCE" ollama pull "$MODELLO"

# --- 4. Esegui il benchmark per i due dataset --------------------------------
run_one() {
  local dataset="$1" out_csv="$2"
  shift 2   # gli argomenti restanti (es. --permutazioni) passano a run.py
  echo ">> Benchmark: $dataset -> out/$out_csv $*"
  apptainer exec $NV_FLAG \
    --bind "$REPO/backend:/work" \
    --bind "$BASE/out:/out" \
    --bind "$BASE/hf_cache:/hf_cache" \
    --env HF_HOME=/hf_cache \
    --pwd /work \
    "$BASE/images/benchmark.sif" \
    python run.py "$dataset" "/out/$out_csv" --modello "$MODELLO" "$@"
}

# Ogni run copre ENTRAMBE le campagne del modello, la corta per prima:
# - BoolQ a parafrasi: split validation completo (3270 domande) x (1 originale
#   + NUM_PARAFRASI parafrasi generate dal modello stesso; 30 dal Test 007,
#   era 10 fino al Test 006 — si cambia in backend/motore/config.py);
# - CommonsenseQA a permutazioni: split validation completo (1221 domande) x
#   120 permutazioni esaustive delle opzioni.
# Percorso unico, tutte le varianti accettate, nessuno scarto (dal Test 002).
#
# Smoke test (1 domanda per dataset, CSV separati):
#   SMOKE=1 bash run_benchmark.sh --<modello>
if [ "${SMOKE:-0}" = "1" ]; then
  run_one boolq          "smoke_boolq_par_${TAG_FILE}.csv"           --num 1
  run_one commonsenseqa  "smoke_commonsenseqa_perm_${TAG_FILE}.csv"  --num 1
else
  run_one boolq          "risultati_boolq_par_${TAG_FILE}.csv"
  run_one commonsenseqa  "risultati_commonsenseqa_perm_${TAG_FILE}.csv"
fi

echo
echo ">> Fatto. CSV in: $BASE/out"
echo ">> RICORDA: copia i CSV via dal cluster (scratch e' volatile), es. da locale:"
echo "   scp 'utente@thor:$BASE/out/'*.csv ."
