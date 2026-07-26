#!/bin/bash
#SBATCH -J ac_b1u1_cG
#SBATCH -p serial_requeue
#SBATCH --array=0-725%24
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -o logs/ac_projected_hf_b1_u1_%A_%a.out
#SBATCH -e logs/ac_projected_hf_b1_u1_%A_%a.err
#SBATCH --requeue

set -euo pipefail

if [[ "${AC_LOCAL_SMOKE:-0}" == "1" ]]; then
  REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
  PYTHON_BIN=${PYTHON_BIN:-"python3"}
  cd "$REPO_ROOT"
else
  module load python/3.12.11-fasrc02
  REPO_ROOT=${REPO_ROOT:-"/n/home06/nchadha/Chiral_DW"}
  cd "$REPO_ROOT"
  source .venv/bin/activate
  PYTHON_BIN=${PYTHON_BIN:-"python"}
fi

mkdir -p logs

# Override any value with:
# sbatch --export=ALL,NAME=value jobs/scan_ac_projected_hf_b1_u1_array.sh
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/ac_b1_u1_cg_taige_dual_gate_nll6_grid11_nk15_30"}

B1_MIN=${B1_MIN:-"-0.1"}
B1_MAX=${B1_MAX:-"0.1"}
N_B1=${N_B1:-"11"}
U1_MIN=${U1_MIN:-"-0.1"}
U1_MAX=${U1_MAX:-"0.1"}
N_U1=${N_U1:-"11"}

N_LL=${N_LL:-"6"}
ACTIVE_BAND=${ACTIVE_BAND:-"0"}
N_K_LIST=${N_K_LIST:-"15,18,21,24,27,30"}

COULOMB_KIND=${COULOMB_KIND:-"dual_gate"}
V0=${V0:-"0.1"}
GATE_DISTANCE=${GATE_DISTANCE:-"2.0"}
Q_MESH=${Q_MESH:-"full"}
Q_SHELL=${Q_SHELL:-"1"}
LOCAL_FIELD_CUTOFF=${LOCAL_FIELD_CUTOFF:-"1"}
EPSILON=${EPSILON:-"16.7"}
GATE_DISTANCE_NM=${GATE_DISTANCE_NM:-"30.0"}
SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.347"}
INCLUDE_Q0=${INCLUDE_Q0:-"1"}
EXCHANGE_SCALE=${EXCHANGE_SCALE:-"1.0"}
HARTREE_SCALE=${HARTREE_SCALE:-"1.0"}
VERTEX_WORKERS=${VERTEX_WORKERS:-"${SLURM_CPUS_PER_TASK:-1}"}
EXCHANGE_WORKERS=${EXCHANGE_WORKERS:-"${SLURM_CPUS_PER_TASK:-1}"}

# Match the native Taige MoTe2 continuum point. The Python driver derives
# a_M and hbar*omega_c from these values and rejects an inconsistent override.
CONTINUUM_THETA_DEG=${CONTINUUM_THETA_DEG:-"3.5"}
CONTINUUM_A0_ANGSTROM=${CONTINUUM_A0_ANGSTROM:-"3.47"}
CONTINUUM_M_EFF=${CONTINUUM_M_EFF:-"0.62"}
MAX_COULOMB_TO_LL_RATIO=${MAX_COULOMB_TO_LL_RATIO:-"0.25"}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N_OCC_PER_K=${N_OCC_PER_K:-"1"}
MAX_ITER=${MAX_ITER:-"800"}
MIN_ITER=${MIN_ITER:-"2"}
MIXING_METHOD=${MIXING_METHOD:-"oda"}
MIXING=${MIXING:-"0.45"}
TOLERANCE=${TOLERANCE:-"1e-8"}
ENERGY_TOLERANCE=${ENERGY_TOLERANCE:-"1e-10"}
FINAL_RESIDUAL_TOLERANCE=${FINAL_RESIDUAL_TOLERANCE:-"1e-7"}
RANDOM_SEED=${RANDOM_SEED:-"1"}

N_THETA=${N_THETA:-"81"}
N_PHI=${N_PHI:-"5"}
PHI_STEP=${PHI_STEP:-"0.2"}
THETA_MIN=${THETA_MIN:-"0.0"}
THETA_MAX=${THETA_MAX:-"3.141592653589793"}

ALLOW_NONCONVERGED_RESPONSE=${ALLOW_NONCONVERGED_RESPONSE:-"0"}

IFS=',' read -r -a N_K_VALUES <<< "$N_K_LIST"
if (( ${#N_K_VALUES[@]} == 0 )); then
  echo "N_K_LIST must contain at least one momentum mesh." >&2
  exit 2
fi

GLOBAL_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
POINTS_PER_MESH=$((N_B1 * N_U1))
TOTAL_TASKS=$((${#N_K_VALUES[@]} * POINTS_PER_MESH))
if (( GLOBAL_TASK_ID < 0 || GLOBAL_TASK_ID >= TOTAL_TASKS )); then
  echo "Task ${GLOBAL_TASK_ID} is outside total size ${TOTAL_TASKS}; exiting."
  exit 0
fi
MESH_INDEX=$((GLOBAL_TASK_ID / POINTS_PER_MESH))
POINT_TASK_ID=$((GLOBAL_TASK_ID % POINTS_PER_MESH))
N_K=${N_K_VALUES[$MESH_INDEX]}
if [[ ! "$N_K" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid n_k value in N_K_LIST: ${N_K}" >&2
  exit 2
fi
BAND_DIAGNOSTICS_N_K=${BAND_DIAGNOSTICS_N_K:-"$N_K"}
MESH_OUTPUT_ROOT="${OUTPUT_ROOT}/nk${N_K}"

COMMON_ARGS=(
  --output-root "$MESH_OUTPUT_ROOT"
  --b1-min "$B1_MIN"
  --b1-max "$B1_MAX"
  --n-b1 "$N_B1"
  --u1-min "$U1_MIN"
  --u1-max "$U1_MAX"
  --n-u1 "$N_U1"
  --n-ll "$N_LL"
  --active-band "$ACTIVE_BAND"
  --n-k "$N_K"
  --band-diagnostics-n-k "$BAND_DIAGNOSTICS_N_K"
  --coulomb-kind "$COULOMB_KIND"
  --v0 "$V0"
  --gate-distance "$GATE_DISTANCE"
  --q-mesh "$Q_MESH"
  --q-shell "$Q_SHELL"
  --local-field-cutoff "$LOCAL_FIELD_CUTOFF"
  --epsilon "$EPSILON"
  --gate-distance-nm "$GATE_DISTANCE_NM"
  --smear-length-nm "$SMEAR_LENGTH_NM"
  --exchange-scale "$EXCHANGE_SCALE"
  --hartree-scale "$HARTREE_SCALE"
  --vertex-workers "$VERTEX_WORKERS"
  --exchange-workers "$EXCHANGE_WORKERS"
  --continuum-theta-deg "$CONTINUUM_THETA_DEG"
  --continuum-a0-angstrom "$CONTINUUM_A0_ANGSTROM"
  --continuum-m-eff "$CONTINUUM_M_EFF"
  --max-coulomb-to-ll-ratio "$MAX_COULOMB_TO_LL_RATIO"
  --n-occ-per-k "$N_OCC_PER_K"
  --max-iter "$MAX_ITER"
  --min-iter "$MIN_ITER"
  --mixing-method "$MIXING_METHOD"
  --mixing "$MIXING"
  --tolerance "$TOLERANCE"
  --energy-tolerance "$ENERGY_TOLERANCE"
  --final-residual-tolerance "$FINAL_RESIDUAL_TOLERANCE"
  --random-seed "$RANDOM_SEED"
  --n-theta "$N_THETA"
  --n-phi "$N_PHI"
  --phi-step "$PHI_STEP"
  --theta-min "$THETA_MIN"
  --theta-max "$THETA_MAX"
)
if [[ "$INCLUDE_Q0" == "0" ]]; then
  COMMON_ARGS+=(--omit-q0)
fi
if [[ "$ALLOW_NONCONVERGED_RESPONSE" == "1" ]]; then
  COMMON_ARGS+=(--allow-nonconverged-response)
fi

if (( POINT_TASK_ID == 0 )); then
  "$PYTHON_BIN" scripts/scan_ac_projected_hf_b1_u1.py \
    "${COMMON_ARGS[@]}" \
    --dry-run
fi

echo "Running physical dual-gate AC task ${GLOBAL_TASK_ID}/${TOTAL_TASKS}: n_k=${N_K} point=${POINT_TASK_ID}/${POINTS_PER_MESH} into ${MESH_OUTPUT_ROOT}"
echo "Resources: cpus=${SLURM_CPUS_PER_TASK:-unset} vertex_workers=${VERTEX_WORKERS} exchange_workers=${EXCHANGE_WORKERS} OMP_NUM_THREADS=${OMP_NUM_THREADS}"
"$PYTHON_BIN" scripts/scan_ac_projected_hf_b1_u1.py \
  "${COMMON_ARGS[@]}" \
  --task-id "$POINT_TASK_ID" \
  --no-write-plan \
  --skip-existing

echo "Task ${GLOBAL_TASK_ID} complete. After one mesh finishes, merge with:"
echo "${PYTHON_BIN} scripts/scan_ac_projected_hf_b1_u1.py --output-root ${MESH_OUTPUT_ROOT} --merge-only"
