# Progetto di Tesi

Tesi SUPSI 2025/26

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

## Dataset utilizzati

| Dataset | ID HuggingFace | Domande totali | Split disponibili | Split usato | Distribuzione classi (split usato) |
| ------- | -------------- | -------------- | ----------------- | ----------- | ---------------------------------- |
| BoolQ (true/false) | `google/boolq` | 12.697 | `train` (9.427), `validation` (3.270) | `validation` | `true` 2.033 (62,2%), `false` 1.237 (37,8%) |
| CommonsenseQA (multiple-choice, 5 opzioni) | `tau/commonsense_qa` | 12.102 | `train` (9.741), `validation` (1.221), `test` (1.140) | `validation` | `A` 239 (19,6%), `B` 255 (20,9%), `C` 241 (19,7%), `D` 251 (20,6%), `E` 235 (19,2%) |

> La distribuzione delle classi si riferisce all'intero split `validation`. Per BoolQ il benchmark ne usa un sottoinsieme di 3.000 domande (vedi sotto), la cui distribuzione è pressoché identica: `true` 1.864 (62,1%), `false` 1.136 (37,9%). Nota lo sbilanciamento di BoolQ verso `true` (~62%) — un modello che rispondesse sempre `true` otterrebbe già quell'accuratezza — mentre CommonsenseQA è quasi uniforme sulle 5 lettere (~20% ciascuna), quindi ogni preferenza sistematica del modello per certe lettere è bias di posizione, non del dataset.

**Perché lo split `validation`.** Per entrambi i dataset le etichette del test set non sono pubbliche: per CommonsenseQA lo split `test` esiste su HuggingFace ma ha `answerKey` vuoto, per BoolQ il test (~3.245 domande del paper originale) non è proprio incluso nella versione HF. Lo split `train` servirebbe al fine-tuning (che qui non facciamo) ed è anche il più esposto a contaminazione nei dati di pre-training dei modelli. `validation` è quindi l'unico split held-out con le risposte note, ed è la convenzione in letteratura per i risultati zero-shot: i numeri restano confrontabili con quelli pubblicati.

**Quante domande per run.** Il motore mescola lo split con seed fisso (`SEED` in `motore/config.py`, riproducibile tra run) e processa `min(NUM_TEST, dimensione dello split)` domande (default `NUM_TEST = 3000`, override con `--num`):

- **BoolQ**: 3.000 domande su 3.270 (sottoinsieme fissato dal seed);
- **CommonsenseQA**: tutte le 1.221 domande di validation (il tetto di 3.000 non viene raggiunto).

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
- Gli indici **`1..N-1`** sono varianti che **mantengono la stessa risposta dell'originale**: se una variante (parafrasi per BoolQ, shuffle delle opzioni per il multiple-choice) cambia idea al modello viene scartata e rigenerata (vedi `scartate_json`), fino a un massimo di `MAX_TENTATIVI_VARIANTE` tentativi. Ogni nuova variante nasce dall'ultima accettata.
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

- **`convergente: false`** — caso eccezionale. Per quella ripetizione il modello ha cambiato risposta su **ogni** variante generata, fino a esaurire i `MAX_TENTATIVI_VARIANTE` tentativi disponibili. Per non lasciare buchi nella lista, l'**ultima** parafrasi tentata viene comunque salvata in `alternative_json`, ma marcata `convergente: false` per segnalare che non è una vera conferma del riferimento ma un *fallback forzato*.

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

Da queste colonne, `generatore_grafici.py` ricostruisce (in modo **dataset-agnostico**: le classi sono rilevate dalle chiavi di `probabilita`, escluso `altro`):

- **Accuratezza globale** Originale / Varianti / Maggioranza → conteggio `risposta_pulita` vs `reale` (la maggioranza è l'`argmax` della somma delle probabilità per classe). Vale per qualsiasi numero di classi.
- **Matrice di confusione** → per i dataset **binari** la classica 2×2 con TP/TN/FP/FN + precision/recall; per il **multiple-choice** una matrice K×(K+1) (riga = classe reale, colonna = classe predetta, ultima colonna `altro` per risposte non valide) come heatmap con diagonale evidenziata, ai tre livelli Originale/Varianti/Maggioranza. I marginali di colonna (stampati a terminale) misurano il **bias di posizione**: quante volte il modello risponde ciascuna lettera. Niente precision/recall per lettera: le lettere sono posizioni, non classi semantiche — la metrica di sintesi resta l'accuratezza.
- **Entropia** (min / max / avg / delta + quartili, scatterplot, Pearson) → dai valori in `probabilita`, con la funzione unica `entropia(probs, classi, includi_altro)`:
  - *senza altro* = entropia sulle sole classi valide (base = K) → per BoolQ coincide con l'entropia **binaria**;
  - *con altro* = entropia su classi valide + `altro` (base = K+1) → per BoolQ coincide con l'entropia **ternaria**.
  Per una multiple-choice a 5 opzioni si ottiene l'entropia su 5 o 6 classi.

> **Nota:** `generatore_grafici.py` legge il campo strutturato `probabilita` prodotto dal motore (`{true, false, altro}` per BoolQ, `{A, B, C, D, E, altro}` per CommonsenseQA) e ci lavora direttamente, senza campi piatti intermedi. Anche il frontend è allineato a `probabilita`.


----

**Esempi utili:**

esempio di parafrasi rifatta perchè risposta differente da quella di domanda originale
![alt text](documentazione/image.png)