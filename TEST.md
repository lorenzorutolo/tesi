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
