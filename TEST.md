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
