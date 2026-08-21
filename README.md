# Progetto di Tesi

Tesi SUPSI 2025/26. Il contesto, la metodologia e i risultati sono nella
documentazione LaTeX (`Documentazione/latex/tesi.tex`): qui c'è solo quello
che serve per orientarsi nel repository e far girare il codice.

---

## Struttura del progetto

```
.
├── backend/                      # Logica Python: motore, CLI, API e analisi
│   ├── motore/                   # Motore di benchmark dataset-agnostico
│   │   ├── benchmark.py          # Ciclo principale + scrittura incrementale del CSV
│   │   ├── config.py             # Costanti di runtime (SEED, NUM_PARAFRASI, modello, endpoint)
│   │   ├── ollama.py             # Chiamate HTTP a Ollama (risposte e parafrasi)
│   │   ├── estrazione.py         # Dai logprobs del primo token alla distribuzione sulle classi
│   │   └── tipi.py               # Dataclass condivise (Domanda, Alternativa, DatasetSpec)
│   ├── specifiche/               # Una spec per dataset (boolq.py, commonsenseqa.py) + registro
│   ├── run.py                    # Entry point CLI: python backend/run.py <dataset>
│   ├── api.py                    # Server HTTP (Flask) per l'interfaccia web
│   ├── generatore_grafici.py     # Analisi delle campagne e figure
│   ├── estrai_matrici.py         # CSV completo -> CSV compatto "matrici"
│   ├── filtra_matrici.py         # Filtro delle varianti stabili sul CSV "matrici"
│   └── requirements.txt          # Dipendenze Python del backend
├── frontend/                     # Interfaccia web React (Vite), gira in locale
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js            # Dev server :5173 + proxy /api -> :8000
│   └── src/
│       ├── main.jsx
│       └── App.jsx
├── cluster/                      # Esecuzione su HPC: definizioni Apptainer e script Slurm
│   └── README.md                 # Istruzioni specifiche del cluster
└── Documentazione/               # Deliverable e sorgenti LaTeX della tesi
    └── latex/                    # tesi.tex, classe SUPSI, bibliografia, immagini
```

Il backend ha **due consumatori dello stesso motore**: la CLI batch
(`run.py`, produce il CSV dei risultati) e il server API (`api.py`, interroga
il modello dal vivo per l'interfaccia web).

---

## Avviare l'interfaccia web

Servono tre componenti in esecuzione.

1. **Ollama** con il modello attivo (default `llama3`, endpoint
   `http://localhost:11434`; entrambi in `backend/motore/config.py`).

2. **Server API** — Flask in ascolto su `http://localhost:8000`:

   ```bash
   pip install -r backend/requirements.txt
   cd backend
   python api.py
   ```

3. **Frontend** — dev server Vite su `http://localhost:5173`:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

Vite inoltra le chiamate `/api/*` al backend su `:8000` (proxy in
`vite.config.js`), quindi il browser parla con una sola origine e non ci sono
problemi di CORS.

L'interfaccia è una chatbox in inglese: si scrive una domanda booleana libera e
si sceglie con quale lettura dell'incertezza calcolare la confidenza.

- **Bayesian (single query)**: una sola interrogazione, confidenza `1 - H[P]`
  sulla distribuzione predittiva. Nessuna ripetizione, perché la probabilità
  viene dai logprob del primo token e non dalla dispersione fra riformulazioni.
- **Credal (repeated queries)**: N riformulazioni della stessa domanda (da 2 a
  30, `MAX_RIPETIZIONI_LIVE` in `api.py`) formano l'insieme credale e la
  confidenza è `1 - AU_C`, con `AU_C = max H[P]` letta sul caso peggiore.

In entrambi i casi l'entropia è ternaria in base 3 (`true`/`false`/`altro`,
regime "con altro"), quindi la confidenza sta in [0, 1]. Il frontend chiama
`POST /api/interroga` con `{"domanda": "<testo>", "ripetizioni": N}`: la prima
ripetizione usa la domanda così com'è, le altre N−1 sono parafrasi generate dal
modello stesso. La risposta contiene un'entry per ripetizione, nello stesso
formato delle alternative del CSV; il calcolo della confidenza avviene nel
frontend, che mostra la sola risposta di maggioranza seguita dal valore di
confidenza e da un semaforo rosso/giallo/verde (verde da 0.67, giallo da 0.34).

---

## Formato dei dati prodotti

Le campagne di benchmark si lanciano dalla CLI, che scrive un CSV per campagna:

```bash
python backend/run.py boolq risultati_boolq_par.csv
```

Il formato di quel CSV (le quattro colonne `id,domanda,reale,alternative_json`)
e la struttura del campo `alternative_json` sono documentati negli **Allegati
della tesi** (`Documentazione/latex/tesi.tex`, sezioni *Formato del CSV dei
risultati* e *Formato del campo `alternative_json`*).

Il formato compatto "matrici" derivato da `backend/estrai_matrici.py`
(`id,matrice,colonna_corretta`) è descritto nel docstring dello script.
