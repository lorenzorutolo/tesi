# Branch `test` — registro dei test riproducibili

Questo branch raccoglie i risultati delle campagne di benchmark. **Convenzione: ogni
commit è un test riproducibile** e contiene tutto ciò che serve a rifarlo:

1. il **CSV dei risultati** nella root;
2. il **log del job Slurm** (`job_<id>.out`) — unica traccia dell'esecuzione, dato che
   lo scratch di Thor è volatile;
3. l'aggiornamento di questo file con la scheda del test (seed, commit del codice,
   comando, hardware, esito);
4. il codice nel commit stesso: il branch parte da `dev`, quindi il motore versionato
   qui è quello che ha prodotto i risultati (a meno di quanto annotato nella scheda).

Per riprodurre un test: checkout del commit, e rieseguire il comando indicato nella
scheda. Nota generale sulla riproducibilità (vedi commento in
`backend/motore/benchmark.py`): con seed fissato le risposte sono deterministiche
(argmax sui logprobs del primo token); resta stocastica solo la generazione delle
parafrasi (BoolQ), quindi i test senza parafrasi sono riproducibili end-to-end.

---

## Test 001 — Campagna permutazioni esaustiva CommonsenseQA

| Campo | Valore |
|---|---|
| Risultati | `risultati_commonsenseqa_perm.csv` (1.221 righe dati, 55 MB) |
| Log | `job_43883.out` (job Slurm **43883** su Thor, partizione `gpu`) |
| Lancio / fine | 2026-07-06 15:32 → 2026-07-07 (completato) |
| Comando | `python run.py commonsenseqa /out/risultati_commonsenseqa_perm.csv --permutazioni` (in container `benchmark.sif`, orchestrato da `cluster/run_benchmark.sh` con `USE_GPU=1`) |
| Modello | `llama3` (8B) via Ollama, backend CUDA |
| Hardware | 1× Tesla V100-PCIE-32GB (gnode, driver 575.57.08, CUDA 12.9) |
| Seed | `SEED = 42` (`backend/motore/config.py`) — fissa shuffle dello split e modulo `random` |
| Dataset | CommonsenseQA, split `validation` completo (1.221 domande) |
| Copertura | Modalità esaustiva: tutte le 5! = 120 permutazioni delle opzioni per ogni domanda, nessun retry né scarto (~146.520 interrogazioni) |
| Codice eseguito | Copia del repo al commit `0c02523` (il motore in questo branch, `de9089b`, è identico a meno di un commento in `benchmark.py`) |
| Riproducibilità | Totale in teoria: nessuna parafrasi, varianti enumerate, risposta = argmax deterministico sui logprobs (unica riserva: non-determinismo floating-point su GPU nei quasi-pareggi) |
| Verifiche | Ultima riga `id=1221` con 120 varianti; header integro; `sacct` COMPLETED |

## Test 002 — Campagna parafrasi BoolQ

| Campo | Valore |
|---|---|
| Risultati | `risultati_boolq_par.csv` (3.270 righe dati, 9,2 MB) |
| Log | `job_43949.out` (job Slurm **43949** su Thor, partizione `gpu`) |
| Lancio / fine | 2026-07-07 13:37 → 2026-07-08 (completato) |
| Comando | `python run.py boolq /out/risultati_boolq_par.csv` (in container `benchmark.sif`, orchestrato da `cluster/run_benchmark.sh` con `USE_GPU=1`) |
| Modello | `llama3` (8B) via Ollama, backend CUDA — sia risposte sia generazione parafrasi |
| Hardware | 1× Tesla V100-PCIE-32GB (gnode, driver 575.57.08, CUDA 12.9) |
| Seed | `SEED = 42` (`backend/motore/config.py`) — fissa shuffle dello split e modulo `random`; parafrasi seedate per chiamata (`SEED*1_000_000 + contatore`, vedi `benchmark.py`) |
| Dataset | BoolQ, split `validation` completo (3.270 domande) |
| Copertura | Percorso unico: per ogni domanda 1 originale + `NUM_PARAFRASI = 10` parafrasi indipendenti dell'originale, tutte registrate senza scarti (35.970 risposte + 32.700 generazioni) |
| Codice eseguito | Copia del repo al commit `61490a1` (il motore in questo commit, merge di `4801c61`, è identico: la differenza è solo uno screenshot rimosso) |
| Riproducibilità | Risposte deterministiche (argmax sui logprobs del primo token, seed fissato); le parafrasi sono riproducibili solo **da server Ollama appena avviato** (condizione documentata in `benchmark.py` e README) — sul cluster è automatica, un'istanza Ollama per job |
| Verifiche | 3.270 righe (id 1–3270 unici); 11 elementi in `alternative_json` per ogni riga; `alternative[0]` = domanda originale; header a 4 colonne; dal log: backend CUDA su V100, "Salvataggio completato" alla domanda 3270/3270 |

## Test 003 — Campagna parafrasi BoolQ con Llama 3.1

> **NB (2026-08-13):** questa campagna BoolQ (10 parafrasi) è **superata dal
> Test 009** (30 parafrasi): i risultati BoolQ@10 non vanno più usati in
> analisi. Il CommonsenseQA di Llama 3.1 (Test 004) resta valido (non dipende
> da `NUM_PARAFRASI`).

| Campo | Valore |
|---|---|
| Risultati | `risultati_boolq_par_llama3.1-8b.csv` (3.270 righe dati, 9,9 MB) |
| Log | `job_44910.out` (job Slurm **44910** su Thor, partizione `gpu`) |
| Lancio / fine | 2026-07-18 18:42 → completato (`sacct` COMPLETED, verificato il 2026-07-20) |
| Comando | `python run.py boolq /out/risultati_boolq_par_llama3.1-8b.csv --modello llama3.1:8b` (in container `benchmark.sif`, orchestrato da `cluster/run_benchmark.sh --nomeModello llama3.1:8b` con `USE_GPU=1`; il tag entra nel nome del CSV sanificato `:`→`-`) |
| Modello | `llama3.1:8b` (Llama 3.1 8B) via Ollama, backend CUDA — sia risposte sia generazione parafrasi (riga `Modello: llama3.1:8b` nel log) |
| Hardware | 1× Tesla V100-PCIE-32GB (gnode, driver 575.57.08, CUDA 12.9) |
| Seed | `SEED = 42` (`backend/motore/config.py`) — fissa shuffle dello split e modulo `random`; parafrasi seedate per chiamata (`SEED*1_000_000 + contatore`, vedi `benchmark.py`) |
| Dataset | BoolQ, split `validation` completo (3.270 domande) |
| Copertura | Percorso unico: per ogni domanda 1 originale + `NUM_PARAFRASI = 10` parafrasi indipendenti dell'originale, tutte registrate senza scarti (35.970 risposte + 32.700 generazioni) |
| Codice eseguito | Il repo su Thor è una copia piatta aggiornata via scp ai file del refactor `bcd2438` (`run.py`, `motore/config.py`, `motore/ollama.py`, script `cluster/`); il motore in questo commit è lo stesso codice |
| Riproducibilità | Come Test 002: risposte deterministiche (argmax sui logprobs del primo token, seed fissato); parafrasi riproducibili solo da server Ollama appena avviato (automatico sul cluster). Smoke test pre-lancio passato (2 domande, mappatura primo token→classi verificata per llama3.1) |
| Verifiche | 3.270 righe (id 1–3270 unici); 11 elementi in `alternative_json` per ogni riga; header a 4 colonne; dal log: backend CUDA su V100, `Modello: llama3.1:8b`, "Salvataggio completato" alla domanda 3270/3270; round-trip matrici 11×3 ok (classi `true,false,altro`) |

## Test 004 — Campagna permutazioni esaustiva CommonsenseQA con Llama 3.1

| Campo | Valore |
|---|---|
| Risultati | `risultati_commonsenseqa_perm_llama3.1-8b.csv` (1.221 righe dati, 54 MB) |
| Log | `job_45180.out` (job Slurm **45180** su Thor, partizione `gpu`) |
| Lancio / fine | 2026-07-21 10:47 → completato (log: "Salvataggio completato" alla domanda 1221/1221; CSV scaricato in locale il 2026-07-24) |
| Comando | `python run.py commonsenseqa /out/risultati_commonsenseqa_perm_llama3.1-8b.csv --modello llama3.1:8b` (in container `benchmark.sif`, orchestrato da `cluster/run_benchmark.sh --nomeModello llama3.1:8b` con `USE_GPU=1`; il tag entra nel nome del CSV sanificato `:`→`-`) |
| Modello | `llama3.1:8b` (Llama 3.1 8B) via Ollama, backend CUDA (riga `Modello: llama3.1:8b` nel log) — solo risposte, nessuna generazione di parafrasi |
| Hardware | 1× Tesla V100-PCIE-32GB (gnode, driver 575.57.08, CUDA 12.9) |
| Seed | `SEED = 42` (`backend/motore/config.py`) — fissa shuffle dello split e modulo `random` |
| Dataset | CommonsenseQA, split `validation` completo (1.221 domande, `tau/commonsense_qa`) |
| Copertura | Modalità esaustiva a percorso unico: tutte le 5! = 120 permutazioni delle opzioni per ogni domanda, tutte registrate senza scarti (146.520 risposte) |
| Codice eseguito | Motore (`run.py`, `motore/`) invariato da `bcd2438`, identico a quello in questo commit. Lo script di lancio `cluster/run_benchmark.sh` all'epoca NON era tracciato in git (`cluster/` in `.gitignore` dal `6213338`): sul cluster è andata via scp la copia locale, con la sola campagna commonsenseqa attiva e selezione modello via `--nomeModello`. `cluster/` è tracciata solo dai commit successivi a questo, con CLI diversa |
| Riproducibilità | Totale in teoria, come Test 001: nessuna parafrasi, varianti enumerate, risposta = argmax deterministico sui logprobs del primo token (unica riserva: non-determinismo floating-point su GPU nei quasi-pareggi). Smoke test pre-lancio SALTATO su scelta dell'utente; verifica manuale dal log che modello e dataset fossero quelli giusti |
| Verifiche | 1.221 righe (id 1–1221 unici, nessun buco né duplicato); 120 elementi in `alternative_json` per ogni riga; header a 4 colonne; `reale` sempre in A–E; somma delle probabilità ≈ 1 per tutte le 146.520 varianti; dal log: backend CUDA su V100, tutte le 1221/1221 domande elaborate, nessun errore |

## Test 005 — Campagne BoolQ parafrasi + CommonsenseQA permutazioni con Qwen2.5-14B-Instruct (job unico)

Primo test con la nuova convenzione (dal commit `14207b5`): **1 job = 1 modello =
entrambe le campagne**, un solo commit con 2 CSV + 1 log + questa scheda.

| Campo | Valore |
|---|---|
| Risultati | `risultati_boolq_par_qwen2.5-14b-instruct.csv` (3.270 righe dati, 9,0 MB) e `risultati_commonsenseqa_perm_qwen2.5-14b-instruct.csv` (1.221 righe dati, 54 MB) |
| Log | `job_45670.out` (job Slurm **45670** su Thor, partizione `gpu`) — unico log per entrambe le campagne, BoolQ prima |
| Lancio / fine | 2026-07-24 14:46 → completato (log: "Salvataggio completato" per entrambi i CSV; scaricati in locale il 2026-07-26) |
| Comando | `sbatch sbatch_benchmark.sh --qwen2.5-14b-instruct` → `cluster/run_benchmark.sh` con `USE_GPU=1` esegue in sequenza `python run.py boolq /out/risultati_boolq_par_qwen2.5-14b-instruct.csv --modello qwen2.5:14b-instruct` e `python run.py commonsenseqa /out/risultati_commonsenseqa_perm_qwen2.5-14b-instruct.csv --modello qwen2.5:14b-instruct` (container `benchmark.sif`; alias CLI = tag con `:`→`-`, lo stesso nei nomi CSV) |
| Modello | `qwen2.5:14b-instruct` (Qwen2.5 14B Instruct) via Ollama, backend CUDA (riga `Modello: qwen2.5:14b-instruct` nel log per entrambe le campagne) — stesso modello per risposte e generazione parafrasi (queste solo per BoolQ) |
| Hardware | 1× Tesla V100-PCIE-32GB (gnode, driver 575.57.08, CUDA 12.9) |
| Seed | `SEED = 42` (`backend/motore/config.py`) — fissa shuffle dello split e modulo `random`; parafrasi seedate per chiamata (`SEED*1_000_000 + contatore`, vedi `benchmark.py`) |
| Dataset | BoolQ, split `validation` completo (3.270 domande) + CommonsenseQA, split `validation` completo (1.221 domande, `tau/commonsense_qa`) |
| Copertura | Percorso unico, nessuno scarto. BoolQ: 1 originale + `NUM_PARAFRASI = 10` parafrasi indipendenti per domanda (35.970 risposte + 32.700 generazioni). CommonsenseQA: tutte le 5! = 120 permutazioni delle opzioni per domanda (146.520 risposte) |
| Codice eseguito | Primo test con `cluster/` tracciata in git: script di lancio del commit `14207b5` (scp su Thor il 2026-07-24), motore (`run.py`, `motore/`) invariato da `bcd2438` — entrambi identici al codice in questo commit |
| Riproducibilità | Come Test 003+004: risposte deterministiche (argmax sui logprobs del primo token, seed fissato); parafrasi BoolQ riproducibili solo da server Ollama appena avviato (automatico sul cluster, un'istanza per job). Smoke test pre-lancio eseguito dall'utente (1 domanda per dataset, sessione interattiva su nodo GPU; non compare in questo log) |
| Verifiche | BoolQ: 3.270 righe (id 1–3270 completi e unici), 11 elementi in `alternative_json` per riga. CommonsenseQA: 1.221 righe (id 1–1221 completi e unici), 120 elementi per riga. Header a 4 colonne su entrambi; somma probabilità per variante in [0,999982; 1,000000] su tutte le 182.490 varianti; dal log: `library=CUDA` su V100, 3270/3270 e 1221/1221 domande elaborate, "Salvataggio completato" ×2, nessun errore né traceback |

## Test 006 — Campagne BoolQ parafrasi + CommonsenseQA permutazioni con Gemma-2-9B (job unico)

> **NB (2026-08-02):** la campagna BoolQ di questo test (10 parafrasi) è
> **superata dal Test 007** (30 parafrasi): i risultati BoolQ@10 non vanno più
> usati in analisi. Il CommonsenseQA resta valido (non dipende da
> `NUM_PARAFRASI`).

Primo test dopo il fix dei tag Gemma (`6efa4a6`): l'alias CLI descrittivo
`--gemma2-9b-instruct` mappa sul tag Ollama reale `gemma2:9b` (su Ollama i tag
di default sono già le varianti instruction-tuned; `gemma2:9b-instruct` non
esiste). L'alias fissa anche il suffisso dei CSV. Prima campagna su A100.

| Campo | Valore |
|---|---|
| Risultati | `risultati_boolq_par_gemma2-9b-instruct.csv` (3.270 righe dati, 9,0 MB) e `risultati_commonsenseqa_perm_gemma2-9b-instruct.csv` (1.221 righe dati, 54 MB) |
| Log | `job_45846.out` (job Slurm **45846** su Thor, partizione `gpu`) — unico log per entrambe le campagne, BoolQ prima |
| Lancio / fine | 2026-07-26 15:13 → completato (log: "Salvataggio completato" per entrambi i CSV; fine tra il 27 e il 28/07, `sacct` non verificato; scaricati in locale il 2026-07-28) |
| Comando | `sbatch sbatch_benchmark.sh --gemma2-9b-instruct` → `cluster/run_benchmark.sh` con `USE_GPU=1` esegue in sequenza `python run.py boolq /out/risultati_boolq_par_gemma2-9b-instruct.csv --modello gemma2:9b` e `python run.py commonsenseqa /out/risultati_commonsenseqa_perm_gemma2-9b-instruct.csv --modello gemma2:9b` (container `benchmark.sif`; qui alias CLI ≠ tag: mappatura nel log, riga `Modello: gemma2:9b (suffisso CSV: gemma2-9b-instruct)`) |
| Modello | `gemma2:9b` (Gemma 2 9B, variante instruction-tuned di default su Ollama) via Ollama, backend CUDA (riga `Modello: gemma2:9b` nel log per entrambe le campagne) — stesso modello per risposte e generazione parafrasi (queste solo per BoolQ) |
| Hardware | 1× NVIDIA A100 80GB PCIe (gnode, driver 575.57.08, CUDA 12.9) — prima campagna su A100 anziché V100 |
| Seed | `SEED = 42` (`backend/motore/config.py`) — fissa shuffle dello split e modulo `random`; parafrasi seedate per chiamata (`SEED*1_000_000 + contatore`, vedi `benchmark.py`) |
| Dataset | BoolQ, split `validation` completo (3.270 domande) + CommonsenseQA, split `validation` completo (1.221 domande, `tau/commonsense_qa`) |
| Copertura | Percorso unico, nessuno scarto. BoolQ: 1 originale + `NUM_PARAFRASI = 10` parafrasi indipendenti per domanda (35.970 risposte + 32.700 generazioni). CommonsenseQA: tutte le 5! = 120 permutazioni delle opzioni per domanda (146.520 risposte) |
| Codice eseguito | Script di lancio del commit `6efa4a6` (fix alias→tag Gemma, scp su Thor il 2026-07-26 prima del lancio), motore (`run.py`, `motore/`) invariato da `bcd2438` — entrambi identici al codice in questo commit |
| Riproducibilità | Come Test 005: risposte deterministiche (argmax sui logprobs del primo token, seed fissato); parafrasi BoolQ riproducibili solo da server Ollama appena avviato (automatico sul cluster, un'istanza per job). Smoke test pre-lancio eseguito dall'utente (tokenizer Gemma incluso, sessione interattiva su nodo GPU; non compare in questo log) |
| Verifiche | BoolQ: 3.270 righe (id 1–3270 completi e unici), 11 elementi in `alternative_json` per riga, somma probabilità per variante in [0,971990; 1,000000]. CommonsenseQA: 1.221 righe (id 1–1221 completi e unici), 120 elementi per riga, somma in [0,885866; 0,999959] (182.490 varianti totali). Header a 4 colonne su entrambi; dal log: `library=CUDA` su A100, 3270/3270 e 1221/1221 domande elaborate, "Salvataggio completato" ×2, nessun errore né traceback (grep con word boundary: "error" matcha "terror" nelle domande) |

## Test 007 — Campagna BoolQ parafrasi a 30 ripetizioni con Gemma-2-9B

Rifacimento del solo BoolQ del Test 006 con `NUM_PARAFRASI = 30` (decisione
del relatore del 2026-07-30, cfr. commit `f90da8b`): **sostituisce il BoolQ@10
del Test 006**, che da qui in poi è deprecato. Prima campagna lanciata con la
variabile `DATASET` di `run_benchmark.sh` (commit `ce6eaaa`) per eseguire un
solo dataset. Il CSV ha lo stesso nome di quello del Test 006 (stesso alias
modello): la versione a 10 parafrasi resta solo nella history.

| Campo | Valore |
|---|---|
| Risultati | `risultati_boolq_par_gemma2-9b-instruct.csv` (3.270 righe dati, 25,2 MB) |
| Log | `job_46476.out` (job Slurm **46476** su Thor, partizione `gpu`) — sola campagna BoolQ. Il primo lancio (job **46475**) è stato cancellato prima di produrre risultati: sul cluster era finita una copia corrotta di `sbatch_benchmark.sh` senza direttive `#SBATCH` (sintomo: log `slurm-<id>.out` invece di `job_<id>.out`); wrapper ripristinato da git e ricaricato via scp prima del rilancio |
| Lancio / fine | 2026-07-31 14:32 → completato (log: "Salvataggio completato"; CSV e log scaricati in locale il 2026-08-02) |
| Comando | `DATASET=boolq sbatch sbatch_benchmark.sh --gemma2-9b-instruct` → `cluster/run_benchmark.sh` con `USE_GPU=1` esegue solo `python run.py boolq /out/risultati_boolq_par_gemma2-9b-instruct.csv --modello gemma2:9b` (container `benchmark.sif`; alias CLI ≠ tag: riga `Modello: gemma2:9b (suffisso CSV: gemma2-9b-instruct, campagne: boolq)` nel log) |
| Modello | `gemma2:9b` (Gemma 2 9B, variante instruction-tuned di default su Ollama) via Ollama, backend CUDA — stesso modello per risposte e generazione parafrasi |
| Hardware | 1× NVIDIA A100 80GB PCIe (gnode, CUDA 12.9) |
| Seed | `SEED = 42` (`backend/motore/config.py`) — fissa shuffle dello split e modulo `random`; parafrasi seedate per chiamata (`SEED*1_000_000 + contatore`, vedi `benchmark.py`) |
| Dataset | BoolQ, split `validation` completo (3.270 domande). CommonsenseQA NON rieseguito: vale quello del Test 006 (le permutazioni non dipendono da `NUM_PARAFRASI`) |
| Copertura | Percorso unico, nessuno scarto: per ogni domanda 1 originale + `NUM_PARAFRASI = 30` parafrasi indipendenti (101.370 risposte + 98.100 generazioni, ~3× il BoolQ del Test 006) |
| Codice eseguito | Script di lancio del commit `ce6eaaa` (variabile `DATASET`, scp su Thor il 2026-07-31), motore invariato da `bcd2438` salvo `NUM_PARAFRASI = 30` in `config.py` (commit `f90da8b`) — identici al codice in questo commit |
| Riproducibilità | Come Test 006: risposte deterministiche (argmax sui logprobs del primo token, seed fissato); parafrasi riproducibili solo da server Ollama appena avviato (automatico sul cluster, un'istanza per job). Smoke test non ripetuto: stessa configurazione del Test 006 già validata (cambia solo `NUM_PARAFRASI`) |
| Verifiche | 3.270 righe (id 1–3270 completi e unici); **31** elementi in `alternative_json` per ogni riga; header a 4 colonne; somma probabilità per variante in [0,971990; 1,000000] su tutte le 101.370 varianti; 0 risposte vuote; chiavi `true/false/altro`; dal log: `library=CUDA` su A100 80GB, "Salvataggio completato", mappatura alias→tag corretta |

## Test 008 — Campagne BoolQ parafrasi (30 ripetizioni) + CommonsenseQA permutazioni con Gemma-3-12B (job unico)

Ultimo modello della tabella multi-modello. Prima campagna di Gemma 3: alias CLI
descrittivo `--gemma3-12b-it` sul tag Ollama reale `gemma3:12b` (su Ollama il tag
di default è già la variante instruction-tuned; `gemma3:12b-it` non esiste).
Job unico su entrambi i dataset, con `NUM_PARAFRASI = 30` già attivo (BoolQ a 31
varianti per domanda, come il Test 007). Lo smoke pre-lancio (job **46385**,
2026-07-31) aveva già validato il tokenizer di gemma3 (31 varianti BoolQ / 120
permutazioni CQA, massa di probabilità ≈ 1, nessuna risposta vuota).

| Campo | Valore |
|---|---|
| Risultati | `risultati_boolq_par_gemma3-12b-it.csv` (3.270 righe dati, 25 MB) e `risultati_commonsenseqa_perm_gemma3-12b-it.csv` (1.221 righe dati, 52 MB) |
| Log | `job_46588.out` (job Slurm **46588** su Thor, partizione `gpu`) — unico log per entrambe le campagne, BoolQ prima |
| Lancio / fine | 2026-08-02 16:17 → completato (log: "Salvataggio completato" per entrambi i CSV; CSV e log scaricati in locale il 2026-08-10) |
| Comando | `sbatch sbatch_benchmark.sh --gemma3-12b-it` → `cluster/run_benchmark.sh` con `USE_GPU=1` esegue in sequenza `python run.py boolq /out/risultati_boolq_par_gemma3-12b-it.csv --modello gemma3:12b` e `python run.py commonsenseqa /out/risultati_commonsenseqa_perm_gemma3-12b-it.csv --modello gemma3:12b` (container `benchmark.sif`; alias CLI ≠ tag: riga `Modello: gemma3:12b (suffisso CSV: gemma3-12b-it, campagne: entrambi)` nel log) |
| Modello | `gemma3:12b` (Gemma 3 12B, variante instruction-tuned di default su Ollama) via Ollama, backend CUDA — stesso modello per risposte e generazione parafrasi (queste solo per BoolQ) |
| Hardware | 1× Tesla V100-PCIE-32GB (gnode, driver 575.57.08, CUDA 12.9) |
| Seed | `SEED = 42` (`backend/motore/config.py`) — fissa shuffle dello split e modulo `random`; parafrasi seedate per chiamata (`SEED*1_000_000 + contatore`, vedi `benchmark.py`) |
| Dataset | BoolQ, split `validation` completo (3.270 domande) + CommonsenseQA, split `validation` completo (1.221 domande, `tau/commonsense_qa`) |
| Copertura | Percorso unico, nessuno scarto. BoolQ: 1 originale + `NUM_PARAFRASI = 30` parafrasi indipendenti per domanda (101.370 risposte + 98.100 generazioni). CommonsenseQA: tutte le 5! = 120 permutazioni delle opzioni per domanda (146.520 risposte) |
| Codice eseguito | Script di lancio con variabile `DATASET` (commit `ce6eaaa`) e motore con `NUM_PARAFRASI = 30` (commit `f90da8b`), invariato da `bcd2438` per il resto — identici al codice in questo commit |
| Riproducibilità | Come Test 007: risposte deterministiche (argmax sui logprobs del primo token, seed fissato); parafrasi BoolQ riproducibili solo da server Ollama appena avviato (automatico sul cluster, un'istanza per job). Smoke test pre-lancio eseguito (job 46385, tokenizer gemma3 validato) |
| Verifiche | BoolQ: 3.270 righe (id 1–3270 completi e unici), **31** elementi in `alternative_json` per riga, 0 JSON malformati, 0 campi vuoti, risposte solo `true/false`. CommonsenseQA: 1.221 righe (id 1–1221 completi e unici), 120 elementi per riga, risposte solo A–E. Header a 4 colonne su entrambi; somma probabilità per variante in [1,000000; 1,000000] (BoolQ) e [1,000000; 1,000001] (CQA) su tutte le 247.890 varianti; dal log: `library=CUDA` su V100-PCIE-32GB, 3270/3270 e 1221/1221 domande elaborate, "Salvataggio completato" ×2, nessun traceback né errore |

## Test 009 — Campagna BoolQ parafrasi a 30 ripetizioni con Llama 3.1-8B

Rifacimento del solo BoolQ del Test 003 con `NUM_PARAFRASI = 30`: **sostituisce
il BoolQ@10 del Test 003** (Llama 3.1), che da qui in poi è deprecato. Il
CommonsenseQA di Llama 3.1 resta quello del Test 004 (le permutazioni non
dipendono da `NUM_PARAFRASI`). Campagna a dataset singolo via variabile
`DATASET` di `run_benchmark.sh`, come il Test 007. Il CSV ha lo stesso nome di
quello del Test 003 (stesso alias modello): la versione a 10 parafrasi resta
solo nella history.

| Campo | Valore |
|---|---|
| Risultati | `risultati_boolq_par_llama3.1-8b.csv` (3.270 righe dati, 27,7 MB) |
| Log | `job_47751.out` (job Slurm **47751** su Thor, partizione `gpu`) — sola campagna BoolQ |
| Lancio / fine | 2026-08-10 23:33 → completato (log: "Salvataggio completato"; CSV e log scaricati in locale il 2026-08-13) |
| Comando | `DATASET=boolq sbatch sbatch_benchmark.sh --llama3.1-8b` → `cluster/run_benchmark.sh` con `USE_GPU=1` esegue solo `python run.py boolq /out/risultati_boolq_par_llama3.1-8b.csv --modello llama3.1:8b` (container `benchmark.sif`; alias CLI = suffisso CSV ma ≠ tag: riga `Modello: llama3.1:8b (suffisso CSV: llama3.1-8b, campagne: boolq)` nel log) |
| Modello | `llama3.1:8b` (Llama 3.1 8B) via Ollama, backend CUDA — stesso modello per risposte e generazione parafrasi |
| Hardware | 1× Tesla V100-PCIE-32GB (gnode, driver 575.57.08, CUDA 12.9) |
| Seed | `SEED = 42` (`backend/motore/config.py`) — fissa shuffle dello split e modulo `random`; parafrasi seedate per chiamata (`SEED*1_000_000 + contatore`, vedi `benchmark.py`) |
| Dataset | BoolQ, split `validation` completo (3.270 domande). CommonsenseQA NON rieseguito: vale quello del Test 004 (le permutazioni non dipendono da `NUM_PARAFRASI`) |
| Copertura | Percorso unico, nessuno scarto: per ogni domanda 1 originale + `NUM_PARAFRASI = 30` parafrasi indipendenti (101.370 risposte + 98.100 generazioni, ~3× il BoolQ del Test 003) |
| Codice eseguito | Script di lancio con variabile `DATASET` (commit `ce6eaaa`) e motore con `NUM_PARAFRASI = 30` (commit `f90da8b`), invariato da `bcd2438` per il resto — identici al codice in questo commit |
| Riproducibilità | Come Test 007/008: risposte deterministiche (argmax sui logprobs del primo token, seed fissato); parafrasi riproducibili solo da server Ollama appena avviato (automatico sul cluster, un'istanza per job). Smoke test non ripetuto: stessa configurazione dei Test 003/005 già validata (cambiano solo `NUM_PARAFRASI` e il dataset singolo) |
| Verifiche | 3.270 righe (id 1–3270 completi e unici); **31** elementi in `alternative_json` per ogni riga; header a 4 colonne; risposte solo `true/false`; 0 JSON malformati, 0 campi vuoti; somma probabilità per variante in [0,888284; 1,000001] su tutte le 101.370 varianti; dal log: `library=CUDA` su V100-PCIE-32GB, 3270/3270 domande elaborate, "Salvataggio completato", nessun traceback né errore |
