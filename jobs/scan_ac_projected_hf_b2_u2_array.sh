#!/bin/bash
#SBATCH -J ac_b2u2_cG
#SBATCH -p serial_requeue
#SBATCH --array=0-120
#SBATCH -t 24:00:00
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -o logs/ac_projected_hf_b2_u2_%A_%a.out
#SBATCH -e logs/ac_projected_hf_b2_u2_%A_%a.err
#SBATCH --requeue

set -euo pipefail

module load python/3.12.11-fasrc02

cd /n/home06/nchadha/Chiral_DW
source .venv/bin/activate

mkdir -p logs

# Override any value with:
# sbatch --export=ALL,NAME=value jobs/scan_ac_projected_hf_b2_u2_array.sh
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/ac_b2_u2_cg_dual_gate_n11_nk12_nll8"}

B1_FIXED=${B1_FIXED:-"0.0"}
U1_FIXED=${U1_FIXED:-"0.0"}
B2_MIN=${B2_MIN:-"-0.3"}
B2_MAX=${B2_MAX:-"0.3"}
N_B2=${N_B2:-"11"}
U2_MIN=${U2_MIN:-"-0.3"}
U2_MAX=${U2_MAX:-"0.3"}
N_U2=${N_U2:-"11"}

N_LL=${N_LL:-"8"}
ACTIVE_BAND=${ACTIVE_BAND:-"0"}
N_K=${N_K:-"12"}
BAND_DIAGNOSTICS_N_K=${BAND_DIAGNOSTICS_N_K:-"9"}

COULOMB_KIND=${COULOMB_KIND:-"dimensionless_dual_gate"}
V0=${V0:-"0.2"}
GATE_DISTANCE=${GATE_DISTANCE:-"2.0"}
Q_SHELL=${Q_SHELL:-"1"}
LOCAL_FIELD_CUTOFF=${LOCAL_FIELD_CUTOFF:-"1"}
EPSILON=${EPSILON:-"16.7"}
GATE_DISTANCE_NM=${GATE_DISTANCE_NM:-"30.0"}
SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.347"}
INCLUDE_Q0=${INCLUDE_Q0:-"1"}
EXCHANGE_SCALE=${EXCHANGE_SCALE:-"1.0"}
HARTREE_SCALE=${HARTREE_SCALE:-"1.0"}
MOIRE_LENGTH_NM=${MOIRE_LENGTH_NM:-"1.0"}
ENERGY_UNIT_MEV=${ENERGY_UNIT_MEV:-"1.0"}

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

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TOTAL_TASKS=$((N_B2 * N_U2))
if (( TASK_ID >= TOTAL_TASKS )); then
  echo "Task ${TASK_ID} is outside mesh size ${TOTAL_TASKS}; exiting."
  exit 0
fi

OMIT_Q0_FLAG=()
if [[ "$INCLUDE_Q0" == "0" ]]; then
  OMIT_Q0_FLAG=(--omit-q0)
fi

NONCONVERGED_FLAG=()
if [[ "$ALLOW_NONCONVERGED_RESPONSE" == "1" ]]; then
  NONCONVERGED_FLAG=(--allow-nonconverged-response)
fi

echo "Running AC projected HF b2/u2 task ${TASK_ID}/${TOTAL_TASKS} into ${OUTPUT_ROOT}"
python scripts/scan_ac_projected_hf_b2_u2.py \
  --output-root "$OUTPUT_ROOT" \
  --b1 "$B1_FIXED" \
  --u1 "$U1_FIXED" \
  --b2-min "$B2_MIN" \
  --b2-max "$B2_MAX" \
  --n-b2 "$N_B2" \
  --u2-min "$U2_MIN" \
  --u2-max "$U2_MAX" \
  --n-u2 "$N_U2" \
  --task-id "$TASK_ID" \
  --n-ll "$N_LL" \
  --active-band "$ACTIVE_BAND" \
  --n-k "$N_K" \
  --band-diagnostics-n-k "$BAND_DIAGNOSTICS_N_K" \
  --coulomb-kind "$COULOMB_KIND" \
  --v0 "$V0" \
  --gate-distance "$GATE_DISTANCE" \
  --q-shell "$Q_SHELL" \
  --local-field-cutoff "$LOCAL_FIELD_CUTOFF" \
  --epsilon "$EPSILON" \
  --gate-distance-nm "$GATE_DISTANCE_NM" \
  --smear-length-nm "$SMEAR_LENGTH_NM" \
  "${OMIT_Q0_FLAG[@]}" \
  --exchange-scale "$EXCHANGE_SCALE" \
  --hartree-scale "$HARTREE_SCALE" \
  --moire-length-nm "$MOIRE_LENGTH_NM" \
  --energy-unit-mev "$ENERGY_UNIT_MEV" \
  --n-occ-per-k "$N_OCC_PER_K" \
  --max-iter "$MAX_ITER" \
  --min-iter "$MIN_ITER" \
  --mixing-method "$MIXING_METHOD" \
  --mixing "$MIXING" \
  --tolerance "$TOLERANCE" \
  --energy-tolerance "$ENERGY_TOLERANCE" \
  --final-residual-tolerance "$FINAL_RESIDUAL_TOLERANCE" \
  --random-seed "$RANDOM_SEED" \
  --n-theta "$N_THETA" \
  --n-phi "$N_PHI" \
  --phi-step "$PHI_STEP" \
  --theta-min "$THETA_MIN" \
  --theta-max "$THETA_MAX" \
  "${NONCONVERGED_FLAG[@]}" \
  --skip-existing

echo "Task ${TASK_ID} complete. After the array finishes, merge with:"
echo "python scripts/scan_ac_projected_hf_b2_u2.py --output-root ${OUTPUT_ROOT} --merge-only"
