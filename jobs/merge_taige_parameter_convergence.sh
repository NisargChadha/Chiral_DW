#!/bin/bash
#SBATCH -J taige_conv_merge
#SBATCH -p serial_requeue
#SBATCH -t 01:00:00
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -o logs/taige_parameter_convergence_merge_%j.out
#SBATCH -e logs/taige_parameter_convergence_merge_%j.err
#SBATCH --requeue

set -euo pipefail

module load python/3.12.11-fasrc02

cd /n/home06/nchadha/Chiral_DW
source .venv/bin/activate

mkdir -p logs

SCAN_AXIS=${SCAN_AXIS:-"plane-wave-shell"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/taige_convergence_plane_wave_shell_theta35_u0"}

python scripts/scan_taige_parameter_convergence.py \
  --scan-axis "$SCAN_AXIS" \
  --output-root "$OUTPUT_ROOT" \
  --merge-only
