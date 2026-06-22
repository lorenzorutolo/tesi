# ProgettoSemestre

Progetto Semestre SUPSI 2025/26

---

## Struttura del progetto

```
.
├── backend/                  # Logica Python: motore, CLI, API e analisi
│   ├── motore/               # Motore di benchmark dataset-agnostico
│   ├── specifiche/           # Una spec per dataset (es. boolq.py) + registro
│   ├── run.py                # Entry point CLI: python backend/run.py <dataset>
│   ├── api.py                # Server HTTP (Flask) per l'interfaccia web
│   ├── requirements.txt      # Dipendenze Python del backend
│   └── generatore_grafici.py # Analisi e grafici (script Colab)
└── frontend/                 # Interfaccia web React (Vite), gira in locale
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        └── App.jsx
```

Il backend ha **due consumatori dello stesso motore**: la CLI batch (genera il CSV per l'analisi offline) e il server API (interroga il modello dal vivo per l'interfaccia web).

### CLI batch (genera il CSV)

```bash
python backend/run.py boolq
```

Il primo argomento è il nome del dataset (vedi `backend/specifiche/`); se omesso usa `boolq`. Genera (o aggiorna) `risultati_benchmark.csv` nella root del progetto, consumato poi da `generatore_grafici.py`.

### Interfaccia web (interrogazione live)

L'interfaccia web non legge più il CSV: interroga il modello in tempo reale tramite un piccolo server HTTP (Flask). Servono tre componenti in esecuzione:

1. **Ollama** con `llama3` attivo (default `http://localhost:11434`).
2. **Server API** (in ascolto su `http://localhost:8000`):

   ```bash
   pip install -r backend/requirements.txt
   cd backend
   python api.py
   ```

3. **Frontend** (dev server Vite su `http://localhost:5173`):

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

Vite inoltra le chiamate `/api/*` al backend su `:8000` (proxy in `vite.config.js`), quindi non serve copiare nessun CSV: al click del pulsante il frontend chiama `POST /api/interroga`, che estrae una domanda casuale e ne calcola originale + ripetizioni dal vivo.

---

## Struttura del CSV

Il file ha esattamente **6 colonne** :

```
id,domanda,reale,num_scartate,alternative_json,scartate_json
```

| Colonna            | Tipo               | Significato                                                                                                  |
| ------------------ | ------------------ | ------------------------------------------------------------------------------------------------------------ |
| `id`               | `int`              | Indice progressivo della domanda (`1..N`)                                                                    |
| `domanda`          | `str`              | Testo della domanda originale dal dataset BoolQ                                                              |
| `reale`            | `"true"`/`"false"` | Risposta corretta di BoolQ                                                                                   |
| `num_scartate`     | `int`              | Numero totale di parafrasi scartate per questa domanda (= `len(scartate_json)`)                              |
| `alternative_json` | JSON (string)      | Lista serializzata delle `RIPETIZIONI_PER_DOMANDA` alternative **accettate** (originale + parafrasi valide)  |
| `scartate_json`    | JSON (string)      | Lista serializzata delle parafrasi **scartate**  |

---

## Struttura di `alternative_json`

È una **lista JSON** lunga `RIPETIZIONI_PER_DOMANDA`.

- L'**indice 0** è sempre la domanda originale ed è quella che **fissa la risposta di riferimento**.
- Gli indici **`1..N-1`** sono parafrasi che **mantengono la stessa risposta dell'originale**: se una parafrasi cambia idea al modello viene scartata e rigenerata (vedi `scartate_json`), fino a un massimo di `MAX_TENTATIVI_PARAFRASI` tentativi. Ogni nuova parafrasi nasce dall'ultima accettata.
- Se per una rep si esauriscono i tentativi senza mai ottenere la stessa risposta, l'ultima parafrasi viene comunque tenuta e marcata con `convergente: false`.

Ogni elemento è un oggetto con 4 campi spiegati di seguito:

```json
{
  "domanda_alt":     "Will additional installments of Bee and PuppyCat be produced?",
  "risposta_pulita": "true",
  "probabilita":     {"true": 0.999817930678194, "false": 0.00017985425916999988, "altro": 1.6143687038687492e-06},
  "convergente":     true
}
```

| Campo             | Significato                                                                                                                                         |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `domanda_alt`     | Testo effettivo presentato al modello in questa ripetizione (originale o parafrasi)                                                                 |
| `risposta_pulita` | Classe vincente decisa con `argmax` sulle classi del dataset (per BoolQ `"true"`/`"false"`, pareggio → `"altro"`)                                   |
| `probabilita`     | Distribuzione `classe → massa di probabilità` sui token top-K. Le chiavi sono le classi dichiarate dal dataset più `"altro"` (sempre presente). Per BoolQ: `true`, `false`, `altro` |
| `convergente`     | Flag che distingue parafrasi stabili da parafrasi che hanno cambiato risposta                         |

> Le chiavi di `probabilita` dipendono dal dataset: per il true/false sono `true`/`false`, per una multiple-choice a 4 opzioni sarebbero `A`/`B`/`C`/`D`. La chiave `altro` raccoglie la massa dei token non riconosciuti.

### Il flag `convergente`

Serve a distinguere due tipi diversi di entry dentro `alternative_json`:

- **`convergente: true`** — caso normale. La parafrasi è stata accettata perché, interrogando il modello, ha prodotto la stessa `risposta_pulita` (`true`/`false`) della domanda originale. Rappresenta quindi una ripetizione *stabile*: stesso significato logico, stesso esito del modello.

- **`convergente: false`** — caso eccezionale. Per quella ripetizione il modello ha cambiato risposta su **ogni** parafrasi generata, fino a esaurire i `MAX_TENTATIVI_PARAFRASI` tentativi disponibili. Per non lasciare buchi nella lista, l'**ultima** parafrasi tentata viene comunque salvata in `alternative_json`, ma marcata `convergente: false` per segnalare che non è una vera conferma del riferimento ma un *fallback forzato*.

---

## Struttura di `scartate_json`

Stessa forma di `alternative_json`: una **lista JSON** di oggetti con i medesimi 4 campi.

Contiene tutte le parafrasi generate durante il retry che **hanno cambiato la risposta** rispetto al riferimento e sono state quindi scartate. 

Nelle scartate il campo `convergente` **non va interpretato**: tutte le entry in `scartate_json` sono per costruzione divergenti dal riferimento — è proprio il motivo per cui sono finite qui

> **Nota:** queste probabilità **non sono normalizzate**. 
>
> Quando serve una distribuzione vera (es. per il calcolo dell'entropia), `generatore_grafici.py` rinormalizza esplicitamente dividendo per la somma.

---

## Cosa viene derivato a tempo di analisi

Da queste colonne, `generatore_grafici.py` ricostruisce:

- **Matrice di confusione "originale"** → da `alternative[0].risposta_pulita` vs `reale`
- **Matrice di confusione "varianti"** → da `alternative[1:].risposta_pulita` vs `reale`
- **Matrice di confusione "maggioranza"** → da `sum(probabilita["true"])` vs `sum(probabilita["false"])` su tutte le alternative
- **Distribuzione voti** (es. `"3-0"`, `"2-1"`) → conteggio dei `risposta_pulita`
- **Entropia binaria / ternaria** (min / max / avg / delta) → dai valori in `probabilita`
- **Quartili di entropia, scatterplot, Pearson** → tutto dai valori in `probabilita`

> **Nota:** `generatore_grafici.py` legge il campo strutturato `probabilita` (`{true, false, altro}`) prodotto dal motore: al caricamento (`_estrai_prob_piatte`) ne ricava i campi piatti `p_true_raw/p_false_raw/p_altri_raw` usati internamente dai grafici. È retro-compatibile con i CSV vecchi. Anche il frontend è allineato a `probabilita`.


----

**Esempi utili:**

esempio di parafrasi rifatta perchè risposta differente da quella di domanda originale
![alt text](documentazione/image.png)