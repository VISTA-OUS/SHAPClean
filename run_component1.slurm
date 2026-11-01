#!/bin/bash
#SBATCH --job-name=c1-poison-audit
#SBATCH --output=logs/c1_%j.out
#SBATCH --error=logs/c1_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Runs one SHAPClean script on SLURM. Pass the conda install folder, the env name and the script path:
#   mkdir -p logs
#   sbatch --export=ALL,CONDA_BASE=<miniconda_dir>,CONDA_ENV=<env_name> run_component1.slurm \
#       scripts/component1_poison_audit/04_poison_rate_sweep.py --seed 8459 --trigger representative
# (logs/ must exist beforehand: SLURM opens the --output file before the script runs and does not create folders.)
# Override --time/--mem at submission when a job needs longer (e.g. sbatch --time=48:00:00 ...).

TARGET_SCRIPT="${1:?Missing argument: path of the script to run}"
shift   # the remaining arguments go to the Python script itself, e.g. ... 04_poison_rate_sweep.py --seed 8459
# Load conda straight from its install folder, NOT through ~/.bashrc: Ubuntu's default .bashrc returns at once in a
# non-interactive shell (which an sbatch job is), so the `conda init` block at its end never runs.
# The env name must be given explicitly: a cluster may hold several envs with different library versions, and a .pkl
# model can only be read by the XGBoost/NumPy versions that wrote it.
CONDA_BASE="${CONDA_BASE:?Missing CONDA_BASE (miniconda install folder) - see how to submit at the top of this file}"
CONDA_ENV="${CONDA_ENV:?Missing CONDA_ENV (env with the libraries installed) - see how to submit at the top of this file}"

set -eo pipefail
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# config.yaml sets core_dist_n_jobs=-1 (hdbscan/joblib) and n_jobs=-1 (XGBoost/OpenMP). Both read the core count of
# the WHOLE NODE, not the share SLURM allocated, so without these variables the job starts too many threads, competes
# for CPU with other jobs and runs SLOWER than it would otherwise.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK}"

# load_config() reads 'configs/config.yaml' by a RELATIVE path -> the job must run from the repository root.
cd "${SLURM_SUBMIT_DIR}"

python -c "import hdbscan, shap, xgboost, sklearn, pandas, numpy, yaml, psutil" || {
  echo "Missing libraries in env '${CONDA_ENV}'. Install them into that env and submit again."
  exit 1
}

echo "[$(date)] START: ${TARGET_SCRIPT} $* (env=${CONDA_ENV}, CPU=${SLURM_CPUS_PER_TASK}, node=${SLURMD_NODENAME})"
python "${TARGET_SCRIPT}" "$@"
echo "[$(date)] DONE: ${TARGET_SCRIPT}"
