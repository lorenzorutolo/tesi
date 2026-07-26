# Esecuzione del benchmark su Thor (Apptainer/Singularity)

Due container separati:

- **`ollama.def`** → server LLM (immagine `ollama/ollama`), avvia `ollama serve`.
- **`benchmark.def`** → ambiente Python con le dipendenze di `backend/requirements.txt`,
  esegue `backend/run.py`.

Il client Python interroga il server su `http://localhost:11434`: funziona perche' Apptainer
condivide la rete dell'host, quindi `localhost` e' lo stesso per entrambi i container.

> **Storage (dalla guida Thor):** `scratch` e' un'area **temporanea** (puo' essere cancellata
> senza preavviso). Tieni `.sif`, modelli, cache e CSV **dentro `~/scratch`** e **copia via i CSV**
> appena pronti. Non e' un fileserver.

## Cheat sheet — ciclo completo di una campagna (comandi pronti)

Host: `lorenzo.rutolo@thor`. I comandi `scp`/`ssh` si lanciano dal terminale
LOCALE (PowerShell, dalla root del repo); quelli `sbatch`/`squeue` da dentro
il cluster (hnode01).

### 1. Carica gli script aggiornati sul cluster (locale)

```powershell
scp cluster/run_benchmark.sh cluster/sbatch_benchmark.sh lorenzo.rutolo@thor:~/scratch/benchmark-thor/repo/cluster/
```

(Serve solo se gli script sono cambiati dall'ultima campagna.)

### 2. Lancia il job (dentro il cluster)

```bash
ssh lorenzo.rutolo@thor
cd ~/scratch/benchmark-thor/repo/cluster
sbatch sbatch_benchmark.sh --<modello>     # es. --qwen2.5-14b-instruct
# -> "Submitted batch job <id>": segnati l'id, serve per log e download
```

Smoke test prima del job lungo (consigliato per un modello mai provato):
vedi la sezione "Test rapido prima del job lungo" piu' sotto.

### 3. Controlla a che punto e' (locale, senza entrare nel cluster)

```powershell
# stato del job: PD = in coda, R = in esecuzione, niente = finito
ssh lorenzo.rutolo@thor "squeue -u lorenzo.rutolo"

# a che domanda e' arrivato (es. "Elaborazione Domanda 812/3270";
# prima campagna = /3270 boolq, seconda = /1221 commonsenseqa)
ssh lorenzo.rutolo@thor "grep 'Elaborazione Domanda' ~/scratch/benchmark-thor/repo/cluster/job_<id>.out | tail -1"

# ultime righe del log (per errori o "Salvataggio completato")
ssh lorenzo.rutolo@thor "tail -30 ~/scratch/benchmark-thor/repo/cluster/job_<id>.out"
```

Da dentro il cluster (dopo `ssh`) il log live si segue con:

```bash
tail -f ~/scratch/benchmark-thor/repo/cluster/job_<id>.out    # Ctrl+C per uscire
```

Percorso COMPLETO obbligatorio: appena entri sei in `~`, ma Slurm scrive il
log nella cartella da cui hai fatto `sbatch`. Se il file non c'e': o il job
e' ancora in coda (`squeue -u $USER`, stato PD), o il percorso e' un altro —
`scontrol show job <id> | grep StdOut` stampa quello vero. Nota: `tail` non
esiste in PowerShell, da locale va sempre avvolto in `ssh "..."`.

### 4. Scarica i risultati a fine job (locale, dalla root del repo)

```powershell
scp "lorenzo.rutolo@thor:~/scratch/benchmark-thor/out/risultati_*_<tag>.csv" .
scp "lorenzo.rutolo@thor:~/scratch/benchmark-thor/repo/cluster/job_<id>.out" .
```

`<tag>` = alias del modello senza `--` (es. `qwen2.5-14b-instruct`,
`gemma2-9b-instruct`): e' il suffisso che `run_benchmark.sh` mette nei nomi
dei CSV; le virgolette servono perche' l'asterisco deve espanderlo il server,
non PowerShell. Scarica SUBITO: lo scratch e' volatile.

## Layout in scratch

Percorso canonico: **`/mnt/beegfs/scratch/<nome.cognome>/benchmark-thor/`** (il percorso
reale; `~/scratch` e' solo un symlink e la guida Thor sconsiglia di usarlo dentro i
container). Nei comandi ssh/scp interattivi il symlink va bene lo stesso.

```
/mnt/beegfs/scratch/<nome.cognome>/benchmark-thor/
├── repo/             # copia di questo progetto
├── images/           # ollama.sif, benchmark.sif
├── ollama_models/    # pesi di llama3 (bind scrivibile)
├── hf_cache/         # cache dataset HuggingFace
└── out/              # CSV prodotti  ->  da copiare via dal cluster
```

## 1. Porta il repo in scratch

Da locale (PowerShell/terminale):

```bash
scp -r C:/Users/loren/Desktop/SUPSI/I3A/tesi/tesi  utente@thor:~/scratch/benchmark-thor/repo
```

(oppure `git clone` direttamente sul nodo, dato che i nodi hanno internet).

## 2. Build delle due immagini (sul nodo, dalla ROOT del repo)

Il path in `benchmark.def` (`backend/requirements.txt`) e' relativo alla cartella da cui lanci
`apptainer build`, quindi **costruisci dalla root del repo**:

```bash
cd ~/scratch/benchmark-thor/repo
apptainer build ../images/ollama.sif     cluster/ollama.def
apptainer build --fakeroot ../images/benchmark.sif cluster/benchmark.def
```

- `ollama.def` non ha `%post` → di solito non serve `--fakeroot`.
- `benchmark.def` fa `pip install` → serve `--fakeroot`. Se non e' abilitato sul nodo, chiedi al
  responsabile del cluster (helpit@supsi.ch) o usa un remote builder.

## 3. Esegui tutto: job Slurm su nodo GPU (via primaria)

hnode01 (dove si atterra con ssh) e' il **headnode**: non ha GPU e serve solo a
sottomettere job. Il benchmark va lanciato come job Slurm sulla partizione `gpu`:

```bash
cd ~/scratch/benchmark-thor/repo/cluster
sbatch sbatch_benchmark.sh --qwen2.5-14b-instruct   # -> "Submitted batch job <id>"
```

Il modello e' obbligatorio e si sceglie con un flag-alias descrittivo (elenco
completo in testa a `run_benchmark.sh`, oppure `--nomeModello TAG` per un tag
arbitrario). Nota: l'alias non coincide sempre col tag Ollama — per Gemma i
tag `*-instruct`/`*-it` non esistono e lo script mappa gli alias sui tag di
default (`gemma2:9b`, `gemma3:12b`), che sono gia' le varianti
instruction-tuned (i base hanno suffisso `-text`). Ogni job esegue
**entrambi** i dataset: BoolQ a parafrasi, poi CommonsenseQA a permutazioni.

Il wrapper richiede 1 GPU (V100/A100), imposta `USE_GPU=1` (flag `--nv` di Apptainer)
e il percorso scratch reale, poi delega a `run_benchmark.sh`. Niente tmux: il job vive
in Slurm e ci si puo' disconnettere.

Monitoraggio e gestione:

```bash
squeue -u $USER                                     # PD = in coda, R = in esecuzione
tail -f job_<id>.out                                # log del benchmark
grep "inference compute" ../../ollama_serve.log     # deve dire library=cuda, NON cpu
scancel <id>                                        # stop (CSV parziale resta valido)
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS   # a fine run: risorse usate davvero
```

Test rapido prima del job lungo (sessione interattiva su nodo GPU, max 30 min):

```bash
# 1. da hnode01: chiedi una shell interattiva su un nodo GPU (resta appeso finche'
#    Slurm non alloca il nodo; il prompt cambia in gnodeXX quando sei dentro)
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash

# 2. sul nodo GPU: stesso filesystem, stessa cartella
cd ~/scratch/benchmark-thor/repo/cluster
USE_GPU=1 SMOKE=1 bash run_benchmark.sh --qwen2.5-14b-instruct
# -> 1 domanda per dataset: out/smoke_boolq_par_<tag>.csv e
#    out/smoke_commonsenseqa_perm_<tag>.csv

# 3. controlli: nvidia-smi mostra la GPU, "inference compute" dice library=cuda,
#    a terminale la massa di probabilita' cade sulle classi (non su "altro"),
#    e i CSV di prova esistono:
head -2 /mnt/beegfs/scratch/$USER/benchmark-thor/out/smoke_*_qwen2.5-14b-instruct.csv

# 4. esci dal nodo (rilascia subito la GPU) e sottometti il job vero
exit
```

### Fallback: esecuzione diretta su CPU (sconsigliata, lentissima)

```bash
cd ~/scratch/benchmark-thor/repo/cluster
bash run_benchmark.sh --<modello>
# percorso custom: BASE=/altro/path bash run_benchmark.sh --<modello>
```

Lo script: avvia Ollama → aspetta che risponda → `ollama pull <tag>` → esegue `run.py`
sulle due campagne (boolq, poi commonsenseqa) → ferma il server.

## 4. Copia i risultati via dal cluster

```bash
# da locale:
scp 'utente@thor:~/scratch/benchmark-thor/out/'*.csv  .
```

## Verifica end-to-end (cosa controllare per dire "funziona")

"End-to-end" = far girare l'intera catena una volta e confermare che esce il risultato giusto.
Non e' un test separato: e' eseguire `run_benchmark.sh` e verificare i punti seguenti.

1. **Server su:** nei log compare `Listening on 127.0.0.1:11434`.
2. **Modello scaricato:** `ollama pull <tag>` arriva a `success`.
3. **Benchmark gira:** a schermo scorrono le domande con le distribuzioni di probabilita'.
4. **Output presente:** i CSV esistono e non sono vuoti:
   ```bash
   ls -l ~/scratch/benchmark-thor/out/
   head -1 ~/scratch/benchmark-thor/out/risultati_boolq_par_<tag>.csv
   ```
   La prima riga deve essere l'header a 4 colonne
   `id,domanda,reale,alternative_json` e sotto una riga di dati per domanda
   (split validation completo; con `SMOKE=1` una sola).
5. **Server fermato:** `apptainer instance list` non mostra piu' `ollama`.

Se i 5 punti sono ok, la pipeline e' verificata. Poi copia i CSV via dal cluster (passo 4).

## Note

- I nodi hanno internet → modello e dataset si scaricano a runtime, nessun pre-staging.
- `NUM_TEST` (numero di domande) e altri parametri sono in `backend/motore/config.py`.
- `hf_cache/` e `ollama_models/` fanno da cache: dal secondo run in poi non si riscarica nulla.
