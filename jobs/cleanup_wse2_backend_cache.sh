#!/bin/bash
#SBATCH -J wse2_ivc_cache_clean
#SBATCH -p serial_requeue
#SBATCH -t 01:00:00
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -o logs/wse2_ivc_cache_clean_%j.out
#SBATCH -e logs/wse2_ivc_cache_clean_%j.err
#SBATCH --requeue

set -euo pipefail

module load python/3.12.11-fasrc02

cd /n/home06/nchadha/Chiral_DW
source .venv/bin/activate

mkdir -p logs

OUTPUT_ROOT=${OUTPUT_ROOT:-"results/wse2_ivc_hysteresis"}
RUN_SLUG=${RUN_SLUG:-"$(basename "${OUTPUT_ROOT%/}")"}
CACHE_BASE_ROOT=${CACHE_BASE_ROOT:-${LAB_SCRATCH_ROOT:-${SCRATCH:-"results/wse2_backend_cache_scratch"}}}
if [[ "$CACHE_BASE_ROOT" != /* ]]; then
  CACHE_BASE_ROOT="$(pwd)/${CACHE_BASE_ROOT}"
fi
CACHE_ROOT=${CACHE_ROOT:-"${CACHE_BASE_ROOT}/${RUN_SLUG}/backend_cache"}
CLEANUP_BACKEND_CACHE=${CLEANUP_BACKEND_CACHE:-"1"}
CLEANUP_DRY_RUN=${CLEANUP_DRY_RUN:-"0"}

DISABLED_FLAG=()
if [[ "$CLEANUP_BACKEND_CACHE" != "1" ]]; then
  DISABLED_FLAG=(--disabled)
fi

DRY_RUN_FLAG=()
if [[ "$CLEANUP_DRY_RUN" == "1" ]]; then
  DRY_RUN_FLAG=(--dry-run)
fi

python scripts/cleanup_taige_backend_cache.py \
  --output-root "$OUTPUT_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --allowed-cache-base-root "$CACHE_BASE_ROOT" \
  "${DISABLED_FLAG[@]}" \
  "${DRY_RUN_FLAG[@]}"
