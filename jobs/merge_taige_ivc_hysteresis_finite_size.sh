#!/bin/bash
#SBATCH -J taige_ivc_hyst_fs_merge
#SBATCH -p serial_requeue
#SBATCH -t 02:00:00
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -o logs/taige_ivc_hyst_fs_merge_%j.out
#SBATCH -e logs/taige_ivc_hyst_fs_merge_%j.err
#SBATCH --requeue

set -euo pipefail

module load python/3.12.11-fasrc02

cd /n/home06/nchadha/Chiral_DW
source .venv/bin/activate

mkdir -p logs

OUTPUT_ROOT=${OUTPUT_ROOT:-"results/taige_ivc_hysteresis_finite_size_nk18_20_grid21"}
N_K_LIST=${N_K_LIST:-"18,19,20"}
MESH_DIR_TEMPLATE=${MESH_DIR_TEMPLATE:-"nk_{n_k:03d}"}
FIT_MIN_CLEAN=${FIT_MIN_CLEAN:-"3"}
FIT_DEGREE=${FIT_DEGREE:-"1"}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/merge_taige_ivc_hysteresis_finite_size.py \
  --output-root "$OUTPUT_ROOT" \
  --n-k-list "$N_K_LIST" \
  --mesh-dir-template "$MESH_DIR_TEMPLATE" \
  --fit-min-clean "$FIT_MIN_CLEAN" \
  --fit-degree "$FIT_DEGREE"
