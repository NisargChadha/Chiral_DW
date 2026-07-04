#!/bin/bash
#SBATCH -J wse2_fs_cG_merge
#SBATCH -p serial_requeue
#SBATCH -t 01:00:00
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -o logs/wse2_finite_size_cG_merge_%j.out
#SBATCH -e logs/wse2_finite_size_cG_merge_%j.err
#SBATCH --requeue

set -euo pipefail

module load python/3.12.11-fasrc02

cd /n/home06/nchadha/Chiral_DW
source .venv/bin/activate

mkdir -p logs

OUTPUT_ROOT=${OUTPUT_ROOT:-"results/wse2_cg_finite_size_nk18_24_u0_15_theta2_4p2"}
FIT_DEGREE=${FIT_DEGREE:-"1"}

python scripts/scan_wse2_finite_size_cg.py \
  --output-root "$OUTPUT_ROOT" \
  --fit-degree "$FIT_DEGREE" \
  --merge-only
