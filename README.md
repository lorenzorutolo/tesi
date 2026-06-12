# ProgettoSemestre

Progetto Semestre SUPSI 2025/26

---

## Struttura del progetto

```
.
├── backend/                  # Logica Python: generazione CSV e analisi
│   ├── generatore_dati.py    # Esegue il benchmark e scrive risultati_benchmark.csv
│   └── generatore_grafici.py # Analisi e grafici (script Colab)
└── frontend/                 # Interfaccia web React (Vite), gira in locale
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        └── App.jsx
```

### Avvio backend

```bash
cd backend
python generatore_dati.py
```

Genera (o aggiorna) `risultati_benchmark.csv` nella root del progetto.

### Avvio frontend (locale)

Il frontend legge `risultati_benchmark.csv` come asset statico da `frontend/public/`. Ogni volta che rigeneri il CSV, copialo lì sopra prima di lanciare il dev server:

```bash
cp risultati_benchmark.csv frontend/public/risultati_benchmark.csv
cd frontend
npm install
npm run dev
```

Il dev server di Vite parte di default su `http://localhost:5173`.

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

Ogni elemento è un oggetto con 6 campi spiegati di seguito:

```json
{
  "domanda_alt":     "Will additional installments of Bee and PuppyCat be produced?",
  "risposta_pulita": "true",
  "p_true_raw":      0.999817930678194,
  "p_false_raw":     0.00017985425916999988,
  "p_altri_raw":     1.6143687038687492e-06,
  "convergente":     true
}
```

| Campo             | Significato                                                                                                                                         |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `domanda_alt`     | Testo effettivo presentato al modello in questa ripetizione (originale o parafrasi)                                                                 |
| `risposta_pulita` | Classificazione `"true"` / `"false"` / `"altro"` decisa con `argmax(p_true_raw, p_false_raw)`                                                       |
| `p_true_raw`      | Massa di probabilità (somma) sui token top-10 che contengono `"true"`                                                                               |
| `p_false_raw`     | Massa di probabilità (somma) sui token top-10 che contengono `"false"`                                                                              |
| `p_altri_raw`     | Massa di probabilità sui restanti token del top-10 (non-true, non-false)                                                                            |
| `convergente`     | Flag che distingue parafrasi stabili da parafrasi che hanno cambiato risposta                         |

### Il flag `convergente`

Serve a distinguere due tipi diversi di entry dentro `alternative_json`:

- **`convergente: true`** — caso normale. La parafrasi è stata accettata perché, interrogando il modello, ha prodotto la stessa `risposta_pulita` (`true`/`false`) della domanda originale. Rappresenta quindi una ripetizione *stabile*: stesso significato logico, stesso esito del modello.

- **`convergente: false`** — caso eccezionale. Per quella ripetizione il modello ha cambiato risposta su **ogni** parafrasi generata, fino a esaurire i `MAX_TENTATIVI_PARAFRASI` tentativi disponibili. Per non lasciare buchi nella lista, l'**ultima** parafrasi tentata viene comunque salvata in `alternative_json`, ma marcata `convergente: false` per segnalare che non è una vera conferma del riferimento ma un *fallback forzato*.

---

## Struttura di `scartate_json`

Stessa forma di `alternative_json`: una **lista JSON** di oggetti con i medesimi 6 campi.

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
- **Matrice di confusione "maggioranza"** → da `sum(p_true_raw)` vs `sum(p_false_raw)` su tutte le alternative
- **Distribuzione voti** (es. `"3-0"`, `"2-1"`) → conteggio dei `risposta_pulita`
- **Entropia binaria / ternaria** (min / max / avg / delta) → dai `p_*_raw`
- **Quartili di entropia, scatterplot, Pearson** → tutto dai `p_*_raw`


----

**Esempi utili:**

esempio di parafrasi rifatta perchè risposta differente da quella di domanda originale
![alt text](documentazione/image.png)