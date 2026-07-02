#!/bin/bash
#SBATCH -J taige_ivc_cache_clean
#SBATCH -p serial_requeue
#SBATCH -t 01:00:00
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -o logs/taige_ivc_cache_clean_%j.out
#SBATCH -e logs/taige_ivc_cache_clean_%j.err
#SBATCH --requeue

set -euo pipefail

module load python/3.12.11-fasrc02

cd /n/home06/nchadha/Chiral_DW
source .venv/bin/activate

mkdir -p logs

OUTPUT_ROOT=${OUTPUT_ROOT:-"results/taige_ivc_hysteresis"}
CACHE_ROOT=${CACHE_ROOT:-"${OUTPUT_ROOT}/backend_cache"}
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
  "${DISABLED_FLAG[@]}" \
  "${DRY_RUN_FLAG[@]}"
