#!/bin/bash
#SBATCH -J taige_ivc_hyst_all
#SBATCH -p serial_requeue
#SBATCH --array=0-83
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -o logs/taige_ivc_hyst_all_%A_%a.out
#SBATCH -e logs/taige_ivc_hyst_all_%A_%a.err
#SBATCH --requeue

set -euo pipefail

module load python/3.12.11-fasrc02

cd /n/home06/nchadha/Chiral_DW
source .venv/bin/activate

mkdir -p logs

OUTPUT_ROOT=${OUTPUT_ROOT:-"results/taige_ivc_hysteresis_nk24_active2_shell5_theta2_4_u0_20"}
CACHE_ROOT=${CACHE_ROOT:-"${OUTPUT_ROOT}/backend_cache"}

U_D_MIN=${U_D_MIN:-"0.0"}
U_D_MAX=${U_D_MAX:-"20.0"}
N_U_D=${N_U_D:-"21"}
THETA_MIN_DEG=${THETA_MIN_DEG:-"2.0"}
THETA_MAX_DEG=${THETA_MAX_DEG:-"4.0"}
N_TWIST=${N_TWIST:-"21"}

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
MAX_ITER=${MAX_ITER:-"800"}
MIN_ITER=${MIN_ITER:-"3"}
MIXING_METHOD=${MIXING_METHOD:-"oda"}
MIXING=${MIXING:-"0.45"}
TOLERANCE=${TOLERANCE:-"1e-8"}
ENERGY_TOLERANCE=${ENERGY_TOLERANCE:-"1e-10"}
FINAL_RESIDUAL_TOLERANCE=${FINAL_RESIDUAL_TOLERANCE:-"1e-7"}
RANDOM_SEEDS=${RANDOM_SEEDS:-"1,7,13,29,53"}
SEED_ORDERED_WEIGHT=${SEED_ORDERED_WEIGHT:-"0.8"}
SEED_RANDOM_WEIGHT=${SEED_RANDOM_WEIGHT:-"0.2"}
INCLUDE_ORDERED_SEED=${INCLUDE_ORDERED_SEED:-"1"}
DEFAULT_RANDOM_SEED=${DEFAULT_RANDOM_SEED:-"7"}

N_THETA=${N_THETA:-"81"}
ENDPOINT_EPS=${ENDPOINT_EPS:-"1e-5"}
DOMAIN_RADIUS=${DOMAIN_RADIUS:-"20.0"}
DOMAIN_WIDTH=${DOMAIN_WIDTH:-"3.0"}
DOMAIN_WINDING=${DOMAIN_WINDING:-"1"}
NAN_TEXTURE_WHEN_IVC_LOWER=${NAN_TEXTURE_WHEN_IVC_LOWER:-"1"}
TEXTURE_ENERGY_TIE_ATOL=${TEXTURE_ENERGY_TIE_ATOL:-"1e-9"}
COMPUTE_INVALID_TEXTURE_CG=${COMPUTE_INVALID_TEXTURE_CG:-"0"}
REQUIRE_CACHE=${REQUIRE_CACHE:-"1"}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TOTAL_TASKS=$((2 * (N_TWIST + N_U_D)))
if (( TASK_ID >= TOTAL_TASKS )); then
  echo "Task ${TASK_ID} is outside combined branch task count ${TOTAL_TASKS}; exiting."
  exit 0
fi

OMIT_Q0_FLAG=()
if [[ "$INCLUDE_Q0" == "0" ]]; then
  OMIT_Q0_FLAG=(--omit-q0)
fi
ORDERED_FLAG=()
if [[ "$INCLUDE_ORDERED_SEED" == "0" ]]; then
  ORDERED_FLAG=(--no-include-ordered-seed)
fi
TEXTURE_FLAG=()
if [[ "$NAN_TEXTURE_WHEN_IVC_LOWER" == "0" ]]; then
  TEXTURE_FLAG=(--allow-texture-in-ivc-ground-state)
fi
DIAGNOSTIC_CG_FLAG=()
if [[ "$COMPUTE_INVALID_TEXTURE_CG" == "1" ]]; then
  DIAGNOSTIC_CG_FLAG=(--compute-invalid-texture-cg)
fi
REQUIRE_CACHE_FLAG=()
if [[ "$REQUIRE_CACHE" == "1" ]]; then
  REQUIRE_CACHE_FLAG=(--require-cache)
fi

echo "Running combined Taige IVC hysteresis linecut task ${TASK_ID}/${TOTAL_TASKS} into ${OUTPUT_ROOT}"
echo "Resources: SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset} VERTEX_WORKERS=${VERTEX_WORKERS} EXCHANGE_WORKERS=${EXCHANGE_WORKERS} DENSITY_VERTEX_RETENTION=${DENSITY_VERTEX_RETENTION} DENSITY_VERTEX_LAYOUT=${DENSITY_VERTEX_LAYOUT} EXCHANGE_REPRESENTATION=${EXCHANGE_REPRESENTATION} FORM_FACTOR_BACKEND=${FORM_FACTOR_BACKEND} SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset} SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-unset}"

python scripts/scan_taige_ivc_hysteresis_linecut.py \
  --output-root "$OUTPUT_ROOT" \
  --cache-root "$CACHE_ROOT" \
  --sweep-axis both \
  --task-id "$TASK_ID" \
  --u-d-min "$U_D_MIN" \
  --u-d-max "$U_D_MAX" \
  --n-u-d "$N_U_D" \
  --theta-min-deg "$THETA_MIN_DEG" \
  --theta-max-deg "$THETA_MAX_DEG" \
  --n-twist "$N_TWIST" \
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
  --final-residual-tolerance "$FINAL_RESIDUAL_TOLERANCE" \
  --random-seeds "$RANDOM_SEEDS" \
  --seed-ordered-weight "$SEED_ORDERED_WEIGHT" \
  --seed-random-weight "$SEED_RANDOM_WEIGHT" \
  "${ORDERED_FLAG[@]}" \
  --default-random-seed "$DEFAULT_RANDOM_SEED" \
  --n-theta "$N_THETA" \
  --endpoint-eps "$ENDPOINT_EPS" \
  --domain-radius "$DOMAIN_RADIUS" \
  --domain-width "$DOMAIN_WIDTH" \
  --domain-winding "$DOMAIN_WINDING" \
  "${TEXTURE_FLAG[@]}" \
  --texture-energy-tie-atol "$TEXTURE_ENERGY_TIE_ATOL" \
  "${DIAGNOSTIC_CG_FLAG[@]}" \
  "${REQUIRE_CACHE_FLAG[@]}"
