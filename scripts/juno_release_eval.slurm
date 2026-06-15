#!/bin/bash
#SBATCH --job-name=argaze_release_eval
#SBATCH --partition=a30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

MODE="${MODE:-smoke}"
DATASET="${DATASET:-egteagaze}"
SUBSET_SIZE="${SUBSET_SIZE:-4}"
RELEASE_ROOT="${RELEASE_ROOT:-/work/jxl220096/projects/argaze/ARGaze-release}"
CKPT_ROOT="${CKPT_ROOT:-/work/jxl220096/projects/argaze/ARGaze-hf-checkpoints}"
PYTHON="${PYTHON:-/work/jxl220096/conda/envs/dinov3/bin/python}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export GLC_SKIP_CUBLAS_PRELOAD="${GLC_SKIP_CUBLAS_PRELOAD:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_argaze_${SLURM_JOB_ID}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONPATH="${RELEASE_ROOT}:${PYTHONPATH:-}"

mkdir -p "${RELEASE_ROOT}/logs"
cd "${RELEASE_ROOT}"

echo "Mode: ${MODE}"
echo "Dataset: ${DATASET}"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"${PYTHON}" - <<'PY'
import os
import torch
print("python cuda visible:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
nvidia-smi || true

case "${DATASET}" in
  egteagaze)
    DATA_ROOT="/home/jxl220096/work/data/egtea"
    CHECKPOINT="${CKPT_ROOT}/egteagaze/best_rollout.pyth"
    ;;
  ego4d)
    DATA_ROOT="/home/jxl220096/work/data/ego4d/v2"
    CHECKPOINT="${CKPT_ROOT}/ego4d/best_rollout.pyth"
    ;;
  egoexo4d)
    DATA_ROOT="/home/jxl220096/work/data/egoexo4d"
    CHECKPOINT="${CKPT_ROOT}/egoexo4d/iid/best_rollout.pyth"
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" == "smoke" ]]; then
  OUT_DIR="output/release_smoke/${DATASET}"
  "${PYTHON}" scripts/eval_shortclip.py \
    --dataset "${DATASET}" \
    --checkpoint "${CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --num-gpus 1 \
    --subset-size "${SUBSET_SIZE}" \
    TEST.BATCH_SIZE 1 \
    DATA_LOADER.NUM_WORKERS 2 \
    WANDB.ENABLE False \
    TENSORBOARD.ENABLE False
elif [[ "${MODE}" == "full" ]]; then
  OUT_DIR="output/release_full/${DATASET}"
  "${PYTHON}" scripts/eval_shortclip.py \
    --dataset "${DATASET}" \
    --checkpoint "${CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --num-gpus 1 \
    TEST.BATCH_SIZE 8 \
    DATA_LOADER.NUM_WORKERS 8 \
    WANDB.ENABLE False \
    TENSORBOARD.ENABLE False
else
  echo "Unsupported MODE=${MODE}" >&2
  exit 2
fi

echo "Output directory: ${OUT_DIR}"
find "${OUT_DIR}" -maxdepth 2 -type f | sort
