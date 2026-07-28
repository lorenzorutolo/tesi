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
