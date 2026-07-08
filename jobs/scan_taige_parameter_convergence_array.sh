#!/bin/bash
#SBATCH -J taige_conv
#SBATCH -p serial_requeue
#SBATCH --array=0-5
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -o logs/taige_parameter_convergence_%A_%a.out
#SBATCH -e logs/taige_parameter_convergence_%A_%a.err
#SBATCH --requeue

set -euo pipefail

module load python/3.12.11-fasrc02

cd /n/home06/nchadha/Chiral_DW
source .venv/bin/activate

mkdir -p logs

# Override any value with:
# sbatch --export=ALL,NAME=value jobs/scan_taige_parameter_convergence_array.sh
SCAN_AXIS=${SCAN_AXIS:-"plane-wave-shell"}
VALUE_LIST=${VALUE_LIST:-"3,4,5,6,7,8"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/taige_convergence_plane_wave_shell_theta35_u0"}
MATERIAL=${MATERIAL:-"mote2"}

U_D=${U_D:-"0.0"}
THETA_DEG=${THETA_DEG:-"3.5"}

N_K=${N_K:-"24"}
PLANE_WAVE_SHELL=${PLANE_WAVE_SHELL:-"5"}
N_BANDS=${N_BANDS:-"2"}
N_ACTIVE_BANDS_PER_VALLEY=${N_ACTIVE_BANDS_PER_VALLEY:-"2"}

Q_MESH=${Q_MESH:-"full"}
Q_SHELL=${Q_SHELL:-"0"}
LOCAL_FIELD_CUTOFF=${LOCAL_FIELD_CUTOFF:-"4"}
INCLUDE_Q0=${INCLUDE_Q0:-"1"}
EPSILON=${EPSILON:-"16.7"}
GATE_DISTANCE_NM=${GATE_DISTANCE_NM:-"30.0"}
SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.347"}
V0=${V0:-"1.0"}
EXCHANGE_SCALE=${EXCHANGE_SCALE:-"1.0"}
HARTREE_SCALE=${HARTREE_SCALE:-"1.0"}
VERTEX_WORKERS=${VERTEX_WORKERS:-"${SLURM_CPUS_PER_TASK:-1}"}
EXCHANGE_WORKERS=${EXCHANGE_WORKERS:-"${SLURM_CPUS_PER_TASK:-1}"}
DENSITY_VERTEX_RETENTION=${DENSITY_VERTEX_RETENTION:-"hartree_only"}
DENSITY_VERTEX_LAYOUT=${DENSITY_VERTEX_LAYOUT:-"auto"}
EXCHANGE_REPRESENTATION=${EXCHANGE_REPRESENTATION:-"auto"}
FORM_FACTOR_BACKEND=${FORM_FACTOR_BACKEND:-"auto"}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N_OCC_PER_K=${N_OCC_PER_K:-"1"}
MAX_ITER=${MAX_ITER:-"100"}
MIN_ITER=${MIN_ITER:-"3"}
MIXING_METHOD=${MIXING_METHOD:-"oda"}
MIXING=${MIXING:-"0.45"}
TOLERANCE=${TOLERANCE:-"1e-8"}
ENERGY_TOLERANCE=${ENERGY_TOLERANCE:-"1e-10"}
SEED_ORDERED_WEIGHT=${SEED_ORDERED_WEIGHT:-"0.8"}
SEED_RANDOM_WEIGHT=${SEED_RANDOM_WEIGHT:-"0.2"}
RANDOM_SEED=${RANDOM_SEED:-"7"}

N_THETA=${N_THETA:-"41"}
ENDPOINT_EPS=${ENDPOINT_EPS:-"1e-5"}
DOMAIN_RADIUS=${DOMAIN_RADIUS:-"20.0"}
DOMAIN_WIDTH=${DOMAIN_WIDTH:-"3.0"}
DOMAIN_WINDING=${DOMAIN_WINDING:-"1"}

COMPUTE_CHERN=${COMPUTE_CHERN:-"1"}
COMPUTE_FINITE_Q_IVC=${COMPUTE_FINITE_Q_IVC:-"0"}
FINITE_Q_SHIFT_POLICY=${FINITE_Q_SHIFT_POLICY:-"exact"}
IVC_BRANCH_POLICY=${IVC_BRANCH_POLICY:-"q0"}
IVC_BRANCH_TIE_ATOL=${IVC_BRANCH_TIE_ATOL:-"1e-9"}
NAN_TEXTURE_WHEN_IVC_LOWER=${NAN_TEXTURE_WHEN_IVC_LOWER:-"1"}
TEXTURE_ENERGY_TIE_ATOL=${TEXTURE_ENERGY_TIE_ATOL:-"1e-9"}
WRITE_HF_PATH_SPECTRA=${WRITE_HF_PATH_SPECTRA:-"0"}
HF_PATH_N_PER_SEGMENT=${HF_PATH_N_PER_SEGMENT:-"36"}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TOTAL_TASKS=$(python -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "$VALUE_LIST")
if (( TASK_ID >= TOTAL_TASKS )); then
  echo "Task ${TASK_ID} is outside scan size ${TOTAL_TASKS}; exiting."
  exit 0
fi

OMIT_Q0_FLAG=()
if [[ "$INCLUDE_Q0" == "0" ]]; then
  OMIT_Q0_FLAG=(--omit-q0)
fi
CHERN_FLAG=()
if [[ "$COMPUTE_CHERN" == "0" ]]; then
  CHERN_FLAG=(--no-chern)
fi
FINITE_Q_IVC_FLAG=()
if [[ "$COMPUTE_FINITE_Q_IVC" == "1" ]]; then
  FINITE_Q_IVC_FLAG=(--compute-finite-q-ivc)
else
  FINITE_Q_IVC_FLAG=(--no-finite-q-ivc)
fi
HF_PATH_FLAG=()
if [[ "$WRITE_HF_PATH_SPECTRA" == "1" ]]; then
  HF_PATH_FLAG=(--write-hf-path-spectra --hf-path-n-per-segment "$HF_PATH_N_PER_SEGMENT")
fi
TEXTURE_FLAG=()
if [[ "$NAN_TEXTURE_WHEN_IVC_LOWER" == "0" ]]; then
  TEXTURE_FLAG=(--allow-texture-in-ivc-ground-state)
fi

echo "Running Taige parameter convergence task ${TASK_ID}/${TOTAL_TASKS} into ${OUTPUT_ROOT}"
echo "Scan: SCAN_AXIS=${SCAN_AXIS} VALUE_LIST=${VALUE_LIST} THETA_DEG=${THETA_DEG} U_D=${U_D}"
echo "Resources: SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset} VERTEX_WORKERS=${VERTEX_WORKERS} EXCHANGE_WORKERS=${EXCHANGE_WORKERS} DENSITY_VERTEX_RETENTION=${DENSITY_VERTEX_RETENTION} DENSITY_VERTEX_LAYOUT=${DENSITY_VERTEX_LAYOUT} EXCHANGE_REPRESENTATION=${EXCHANGE_REPRESENTATION} FORM_FACTOR_BACKEND=${FORM_FACTOR_BACKEND} SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset} SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-unset}"
python scripts/scan_taige_parameter_convergence.py \
  --scan-axis "$SCAN_AXIS" \
  --value-list "$VALUE_LIST" \
  --material "$MATERIAL" \
  --output-root "$OUTPUT_ROOT" \
  --u-d "$U_D" \
  --theta-deg "$THETA_DEG" \
  --task-id "$TASK_ID" \
  --n-k "$N_K" \
  --plane-wave-shell "$PLANE_WAVE_SHELL" \
  --n-bands "$N_BANDS" \
  --n-active-bands-per-valley "$N_ACTIVE_BANDS_PER_VALLEY" \
  --q-mesh "$Q_MESH" \
  --q-shell "$Q_SHELL" \
  --local-field-cutoff "$LOCAL_FIELD_CUTOFF" \
  "${OMIT_Q0_FLAG[@]}" \
  --epsilon "$EPSILON" \
  --gate-distance-nm "$GATE_DISTANCE_NM" \
  --smear-length-nm "$SMEAR_LENGTH_NM" \
  --v0 "$V0" \
  --exchange-scale "$EXCHANGE_SCALE" \
  --hartree-scale "$HARTREE_SCALE" \
  --vertex-workers "$VERTEX_WORKERS" \
  --exchange-workers "$EXCHANGE_WORKERS" \
  --density-vertex-retention "$DENSITY_VERTEX_RETENTION" \
  --density-vertex-layout "$DENSITY_VERTEX_LAYOUT" \
  --exchange-representation "$EXCHANGE_REPRESENTATION" \
  --form-factor-backend "$FORM_FACTOR_BACKEND" \
  --n-occ-per-k "$N_OCC_PER_K" \
  --max-iter "$MAX_ITER" \
  --min-iter "$MIN_ITER" \
  --mixing-method "$MIXING_METHOD" \
  --mixing "$MIXING" \
  --tolerance "$TOLERANCE" \
  --energy-tolerance "$ENERGY_TOLERANCE" \
  --seed-ordered-weight "$SEED_ORDERED_WEIGHT" \
  --seed-random-weight "$SEED_RANDOM_WEIGHT" \
  --random-seed "$RANDOM_SEED" \
  --n-theta "$N_THETA" \
  --endpoint-eps "$ENDPOINT_EPS" \
  --domain-radius "$DOMAIN_RADIUS" \
  --domain-width "$DOMAIN_WIDTH" \
  --domain-winding "$DOMAIN_WINDING" \
  "${CHERN_FLAG[@]}" \
  "${FINITE_Q_IVC_FLAG[@]}" \
  --finite-q-shift-policy "$FINITE_Q_SHIFT_POLICY" \
  --ivc-branch-policy "$IVC_BRANCH_POLICY" \
  --ivc-branch-tie-atol "$IVC_BRANCH_TIE_ATOL" \
  "${TEXTURE_FLAG[@]}" \
  --texture-energy-tie-atol "$TEXTURE_ENERGY_TIE_ATOL" \
  "${HF_PATH_FLAG[@]}" \
  --skip-existing

echo "Task ${TASK_ID} complete. After all tasks finish, merge with:"
echo "sbatch --export=ALL,OUTPUT_ROOT=${OUTPUT_ROOT} jobs/merge_taige_parameter_convergence.sh"
