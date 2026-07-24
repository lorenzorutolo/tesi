#!/bin/bash
# Wrapper Slurm per lanciare il benchmark su un nodo GPU di Thor.
#
# Uso (da hnode01, dentro repo/cluster):
#   sbatch sbatch_benchmark.sh --<modello>
# es.:
#   sbatch sbatch_benchmark.sh --qwen2.5-14b-instruct
# (alias modello e tag disponibili: vedi run_benchmark.sh, a cui gli
# argomenti passano intatti; ogni job esegue ENTRAMBI i dataset)
#
# Monitoraggio:
#   squeue -u $USER                  # PD = in coda, R = in esecuzione
#   tail -f job_<id>.out             # log del benchmark
#   scancel <id>                     # interruzione (il CSV parziale resta valido)
#
#SBATCH --job-name=bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10-00:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Partizione `gpu` (V100 32GB / A100 80GB, max 14 giorni): anche un 14B
# quantizzato sta comodo su entrambe. Niente `gpu_HIGH`: schedula prima ma
# taglia a 2 giorni, troppo poco. Il limite a 10 giorni copre le DUE campagne
# in un job unico anche con modelli ~14B (le singole campagne con un 8B hanno
# preso 1-3 giorni l'una).
#
# Niente srun interno: con --nodes=1 --ntasks=1 l'intero script gira sul nodo
# GPU allocato, e il pattern "istanza Apptainer in background + exec ripetuti"
# di run_benchmark.sh non si presta a step srun separati.

# Percorso scratch REALE (mai il symlink ~/scratch dentro i container).
export BASE=/mnt/beegfs/scratch/$USER/benchmark-thor
export USE_GPU=1

# $0 qui è la copia spool dello script fatta da Slurm: per trovare
# run_benchmark.sh serve la cartella da cui è stato lanciato sbatch.
# Gli argomenti (es. --nomeModello) passano intatti a run_benchmark.sh.
bash "$SLURM_SUBMIT_DIR/run_benchmark.sh" "$@"
