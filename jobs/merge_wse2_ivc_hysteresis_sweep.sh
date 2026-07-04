#!/bin/bash
#SBATCH -J wse2_ivc_hyst_merge
#SBATCH -p serial_requeue
#SBATCH -t 01:00:00
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -o logs/wse2_ivc_hyst_merge_%j.out
#SBATCH -e logs/wse2_ivc_hyst_merge_%j.err
#SBATCH --requeue

set -euo pipefail

module load python/3.12.11-fasrc02

cd /n/home06/nchadha/Chiral_DW
source .venv/bin/activate

mkdir -p logs

OUTPUT_ROOT=${OUTPUT_ROOT:-"results/wse2_ivc_hysteresis_nk24_active2_shell5_theta2_4_u0_20"}
CACHE_ROOT=${CACHE_ROOT:-"${OUTPUT_ROOT}/backend_cache"}
N_OCC_PER_K=${N_OCC_PER_K:-"1"}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/merge_wse2_ivc_hysteresis_sweep.py \
  --output-root "$OUTPUT_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --n-occ-per-k "$N_OCC_PER_K"
