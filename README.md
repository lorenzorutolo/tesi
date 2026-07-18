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

Vite inoltra le chiamate `/api/*` al backend su `:8000` (proxy in `vite.config.js`), quindi non serve copiare nessun CSV: al click del pulsante il frontend chiama `POST /api/interroga`, che estrae una domanda casuale e la interroga dal vivo su un sottoinsieme delle sue varianti (limitato per reattività, vedi `MAX_VARIANTI_LIVE` in `api.py`).

---

## Dataset utilizzati

| Dataset | ID HuggingFace | Domande totali | Split disponibili | Split usato | Distribuzione classi (split usato) |
| ------- | -------------- | -------------- | ----------------- | ----------- | ---------------------------------- |
| BoolQ (true/false) | `google/boolq` | 12.697 | `train` (9.427), `validation` (3.270) | `validation` | `true` 2.033 (62,2%), `false` 1.237 (37,8%) |
| CommonsenseQA (multiple-choice, 5 opzioni) | `tau/commonsense_qa` | 12.102 | `train` (9.741), `validation` (1.221), `test` (1.140) | `validation` | `A` 239 (19,6%), `B` 255 (20,9%), `C` 241 (19,7%), `D` 251 (20,6%), `E` 235 (19,2%) |

> La distribuzione delle classi si riferisce all'intero split `validation`. Per BoolQ il benchmark ne usa un sottoinsieme di 3.000 domande (vedi sotto), la cui distribuzione è pressoché identica: `true` 1.864 (62,1%), `false` 1.136 (37,9%). Nota lo sbilanciamento di BoolQ verso `true` (~62%) — un modello che rispondesse sempre `true` otterrebbe già quell'accuratezza — mentre CommonsenseQA è quasi uniforme sulle 5 lettere (~20% ciascuna), quindi ogni preferenza sistematica del modello per certe lettere è bias di posizione, non del dataset.

**Perché lo split `validation`.** Per entrambi i dataset le etichette del test set non sono pubbliche: per CommonsenseQA lo split `test` esiste su HuggingFace ma ha `answerKey` vuoto, per BoolQ il test (~3.245 domande del paper originale) non è proprio incluso nella versione HF. Lo split `train` servirebbe al fine-tuning (che qui non facciamo) ed è anche il più esposto a contaminazione nei dati di pre-training dei modelli. `validation` è quindi l'unico split held-out con le risposte note, ed è la convenzione in letteratura per i risultati zero-shot: i numeri restano confrontabili con quelli pubblicati.

**Come funziona una run.** Percorso unico per tutti i dataset: il motore mescola lo split con seed fisso (`SEED` in `motore/config.py`) e, per ogni domanda, interroga il modello su **tutte le varianti** fornite dalla spec, una volta ciascuna, **tutte accettate** — niente risposta di riferimento, niente retry, niente scarti a runtime (eventuali scarti si fanno a tempo di analisi del CSV). Default: tutte le domande dello split (override con `--num`):

- **BoolQ**: 3.270 domande × (1 originale + `NUM_PARAFRASI = 10` parafrasi). Le parafrasi sono generate dal modello **sempre a partire dall'originale** (indipendenti, non a catena) con un **seed Ollama deterministico per chiamata** (derivato da `SEED`): stessa run → stesse parafrasi.
- **CommonsenseQA**: 1.221 domande × tutte le 5! = 120 permutazioni dell'ordine delle opzioni (enumerate, originale per prima).

Con `SEED` fissato la run è quindi riproducibile end-to-end: varianti deterministiche e risposte deterministiche (argmax sui logprobs del primo token, indipendente dalla temperatura). Per le parafrasi la condizione (verificata empiricamente) è rieseguire **da server Ollama appena avviato** — la cache dei prompt di richieste precedenti può alterare la generazione a parità di seed — oltre alla parità di versione/hardware; sul cluster è la condizione naturale, dato che ogni job avvia la propria istanza. Riserva teorica residua: non-determinismo floating-point su GPU nei quasi-pareggi.

---

## Struttura del CSV

Il file ha esattamente **4 colonne** :

```
id,domanda,reale,alternative_json
```

| Colonna            | Tipo               | Significato                                                                                                  |
| ------------------ | ------------------ | ------------------------------------------------------------------------------------------------------------ |
| `id`               | `int`              | Indice progressivo della domanda (`1..N`)                                                                    |
| `domanda`          | `str`              | Testo della domanda originale dal dataset                                                                    |
| `reale`            | `str`              | Classe corretta (`"true"`/`"false"` per BoolQ, lettera per il multiple-choice)                               |
| `alternative_json` | JSON (string)      | Lista serializzata di **tutte** le varianti interrogate (originale + parafrasi/permutazioni, nessuna esclusa) |

> I CSV prodotti prima del Test 002 hanno due colonne in più (`num_scartate`, `scartate_json`, eredità della vecchia logica di convergenza) e un campo `convergente` dentro le alternative: `generatore_grafici.py` legge le colonne per nome, quindi restano leggibili.

---

## Struttura di `alternative_json`

È una **lista JSON** con una entry per variante interrogata (11 per BoolQ, 120 per CommonsenseQA).

- L'**indice 0** è sempre la domanda originale.
- Gli indici successivi sono le varianti (parafrasi per BoolQ, permutazioni per il multiple-choice), **tutte registrate qualunque sia la risposta**: se una variante fa cambiare idea al modello resta nel CSV — decidere come trattare questi *flip* è compito dell'analisi, non del motore.

Ogni elemento è un oggetto con 4 campi spiegati di seguito:

```json
{
  "domanda_alt":     "Will additional installments of Bee and PuppyCat be produced?",
  "risposta_pulita": "true",
  "probabilita":     {"true": 0.999817930678194, "false": 0.00017985425916999988, "altro": 1.6143687038687492e-06},
  "ordine":          null
}
```

| Campo             | Significato                                                                                                                                         |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `domanda_alt`     | Testo effettivo presentato al modello in questa variante (originale o parafrasi)                                                                    |
| `risposta_pulita` | Classe vincente decisa con `argmax` sulle classi del dataset (per BoolQ `"true"`/`"false"`, pareggio → `"altro"`)                                   |
| `probabilita`     | Distribuzione `classe → massa di probabilità` sui token top-K. Le chiavi sono le classi dichiarate dal dataset più `"altro"` (sempre presente). Per BoolQ: `true`, `false`, `altro` |
| `ordine`          | Solo multiple-choice: lettere canoniche nell'ordine mostrato al modello (identifica la permutazione). `null` per i dataset senza opzioni (BoolQ)     |

> Le chiavi di `probabilita` dipendono dal dataset: per il true/false sono `true`/`false`, per una multiple-choice a 4 opzioni sarebbero `A`/`B`/`C`/`D`. La chiave `altro` raccoglie la massa dei token non riconosciuti.

> **Nota:** queste probabilità **non sono normalizzate**. 
>
> Quando serve una distribuzione vera (es. per il calcolo dell'entropia), `generatore_grafici.py` rinormalizza esplicitamente dividendo per la somma.

---

## CSV delle matrici (formato compatto derivato)

Dal CSV completo si può derivare un CSV compatto — **una riga per domanda**, tre colonne — con `backend/estrai_matrici.py`:

```bash
python backend/estrai_matrici.py risultati_boolq_par.csv   # -> matrici_boolq_par.csv
```

| Colonna             | Tipo    | Significato                                                                                                   |
| ------------------- | ------- | -------------------------------------------------------------------------------------------------------------- |
| `id`                | `int`   | Id della domanda                                                                                                |
| `matrice`           | `str`   | Matrice delle distribuzioni come lista annidata JSON: una riga interna per ripetizione (originale per prima, poi le varianti nell'ordine del CSV sorgente: 11 per BoolQ, 120 per CommonsenseQA), una colonna interna per classe con `altro` per ultima (BoolQ `true,false,altro`; CommonsenseQA `A,B,C,D,E,altro`) |
| `colonna_corretta`  | `int`   | Indice **0-based** della colonna della matrice corrispondente alla risposta gold                               |

Esempio (BoolQ, colonne interne `true,false,altro`):

```
id,matrice,colonna_corretta
1,"[[1.54692200e-07,9.99995159e-01,4.89459314e-06],[2.03715419e-07,9.99996386e-01,3.40010515e-06],...]",1
2,"[[9.99743308e-01,2.55966282e-04,7.77698812e-07],[9.59023923e-01,4.09692347e-02,4.87604104e-06],...]",0
```

Rilettura in analisi (la cella `matrice` è JSON valido):

```python
df = pd.read_csv("matrici_boolq_par.csv")
M = np.array(json.loads(df.loc[df["id"] == 1, "matrice"].iloc[0]))   # matrice della domanda 1
cc = int(df.loc[df["id"] == 1, "colonna_corretta"].iloc[0])
p_gold = M[:, cc]                                  # prob. della gold per ripetizione
```

I valori hanno 9 cifre significative (errore relativo ~1e-9, irrilevante per KL/entropia) e, come nel CSV sorgente, **non sono normalizzati**.

**Variante per Excel italiano.** Excel con impostazioni italiane usa `;` come separatore di elenco, quindi il doppio clic sul CSV standard mostra tutto in colonna A. Il flag `--excel` genera una variante già apribile col doppio clic — separatore `;` tra le colonne, suffisso `_excel` nel nome; la cella della matrice resta identica (è testo, non numeri, quindi il punto decimale non disturba Excel e la stringa resta JSON valido):

```bash
python backend/estrai_matrici.py risultati_boolq_par.csv --excel   # -> matrici_boolq_par_excel.csv
```

In analisi va letta con `pd.read_csv(..., sep=";")`; per pandas/numpy usare il formato di default.

Il file è una vista derivata e rigenerabile: testi delle parafrasi, `ordine` delle permutazioni e `risposta_pulita` restano solo nel CSV completo (la risposta argmax si ricalcola comunque con `M.argmax(axis=1)`).

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


