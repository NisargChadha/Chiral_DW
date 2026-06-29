#!/bin/bash
#SBATCH -J taige_fs_cG_merge
#SBATCH -p serial_requeue
#SBATCH -t 01:00:00
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -o logs/taige_finite_size_cG_merge_%j.out
#SBATCH -e logs/taige_finite_size_cG_merge_%j.err
#SBATCH --requeue

set -euo pipefail

module load python/3.12.11-fasrc02

cd /n/home06/nchadha/Chiral_DW
source .venv/bin/activate

mkdir -p logs

OUTPUT_ROOT=${OUTPUT_ROOT:-"results/taige_cg_finite_size_nk12_20_u0_theta3p5"}
FIT_DEGREE=${FIT_DEGREE:-"1"}

python scripts/scan_taige_finite_size_cg.py \
  --output-root "$OUTPUT_ROOT" \
  --fit-degree "$FIT_DEGREE" \
  --merge-only
