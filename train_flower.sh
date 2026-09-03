#!/bin/bash
#SBATCH --job-name=grouped-f
#SBATCH --partition=sara
#SBATCH --account=team_sara
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32000
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Diretório de onde `sbatch` foi chamado — mais robusto que fixar o caminho
# na mão, contanto que você sempre rode `sbatch` de dentro do projeto.
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

source .venv/bin/activate

ALLOCATED_CPUS="${SLURM_CPUS_PER_TASK:-20}"
JOB_ID="${SLURM_JOB_ID:-0}"

# Cada ClientApp recebe duas CPUs.
# Isso evita que PyTorch utilize mais threads do que o reservado.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1

# Cache persistente: evita rebaixar datasets do Hugging Face a cada job.
# Vale para tarefas que usam `datasets.load_dataset` sem streaming (ex:
# cifar10, via flwr_datasets). O "stackexchange" NÃO usa esse cache — ele
# faz streaming e usa seu próprio cache em disco
# (pytorchexample/tasks/.stackexchange_cache/), gerado uma vez com
# `prefetch_stackexchange.py` (ver passo 3 acima) antes de rodar sem internet.
export HF_HOME="${HOME}/.cache/custom_fed/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${HOME}/.cache/custom_fed/torch"

mkdir -p "${HF_DATASETS_CACHE}" "${TORCH_HOME}"

# Arquivos temporários do Flower e Ray ficam no nó de cálculo.
JOB_TMP="${SLURM_TMPDIR:-/tmp}/custom_fed_${USER}_${JOB_ID}"
mkdir -p "${JOB_TMP}/flower" "${JOB_TMP}/ray" "${JOB_TMP}"

export FLWR_HOME="${JOB_TMP}/flower"
export RAY_TMPDIR="${JOB_TMP}/ray"
export TMPDIR="${JOB_TMP}"
export TMP="${JOB_TMP}"
export TEMP="${JOB_TMP}"

# Evita conflito de porta caso dois jobs do mesmo usuário caiam no mesmo nó.
export FLWR_LOCAL_CONTROL_API_PORT="$((20000 + JOB_ID % 30000))"

echo "Job ID: ${JOB_ID}"
echo "Partição: ${SLURM_JOB_PARTITION:-?}"
echo "Diretório de submissão: ${SLURM_SUBMIT_DIR}"
echo "Nó: $(hostname)"
echo "CPUs reservadas: ${ALLOCATED_CPUS}"
echo "Python: $(python --version)"
echo "Flower: $(flwr --version)"

# num-supernodes=100 deve bater com num-partitions no pyproject.toml.
# client-resources-num-cpus=2: cada cliente simulado usa 2 CPUs, então até
# ALLOCATED_CPUS/2 clientes treinam ao mesmo tempo por rodada — com
# fraction-train=1.0 (100 clientes por rodada) e ALLOCATED_CPUS=20, isso é
# 10 clientes por vez, em lotes, até completar os 100 na rodada.
 ^XESC...skipping...
# faz streaming e usa seu próprio cache em disco
# (pytorchexample/tasks/.stackexchange_cache/), gerado uma vez com
# `prefetch_stackexchange.py` (ver passo 3 acima) antes de rodar sem internet.
export HF_HOME="${HOME}/.cache/custom_fed/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${HOME}/.cache/custom_fed/torch"

mkdir -p "${HF_DATASETS_CACHE}" "${TORCH_HOME}"

# Arquivos temporários do Flower e Ray ficam no nó de cálculo.
JOB_TMP="${SLURM_TMPDIR:-/tmp}/custom_fed_${USER}_${JOB_ID}"
mkdir -p "${JOB_TMP}/flower" "${JOB_TMP}/ray" "${JOB_TMP}"

export FLWR_HOME="${JOB_TMP}/flower"
export RAY_TMPDIR="${JOB_TMP}/ray"
export TMPDIR="${JOB_TMP}"
export TMP="${JOB_TMP}"
export TEMP="${JOB_TMP}"

# Evita conflito de porta caso dois jobs do mesmo usuário caiam no mesmo nó.
export FLWR_LOCAL_CONTROL_API_PORT="$((20000 + JOB_ID % 30000))"

echo "Job ID: ${JOB_ID}"
echo "Partição: ${SLURM_JOB_PARTITION:-?}"
echo "Diretório de submissão: ${SLURM_SUBMIT_DIR}"
echo "Nó: $(hostname)"
echo "CPUs reservadas: ${ALLOCATED_CPUS}"
echo "Python: $(python --version)"
echo "Flower: $(flwr --version)"

# num-supernodes=100 deve bater com num-partitions no pyproject.toml.
# client-resources-num-cpus=2: cada cliente simulado usa 2 CPUs, então até
# ALLOCATED_CPUS/2 clientes treinam ao mesmo tempo por rodada — com
# fraction-train=1.0 (100 clientes por rodada) e ALLOCATED_CPUS=20, isso é
# 10 clientes por vez, em lotes, até completar os 100 na rodada.
flwr run . --stream \
    --federation-config="num-supernodes=20 client-resources-num-cpus=2 client-resources-num-gpus=0 init-args-num-cpus=${ALLOCATED_CPUS} init-args-num-gpus=0"
