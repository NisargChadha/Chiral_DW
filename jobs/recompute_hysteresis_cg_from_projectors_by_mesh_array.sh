#!/bin/bash
# Recompute linear-interaction cG from stored hysteresis projectors for one mesh.

#SBATCH -J hyst_recompute_cg
#SBATCH -p serial_requeue
#SBATCH --array=0-6
#SBATCH -t 24:00:00
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -o logs/hyst_recompute_cg_%A_%a.out
#SBATCH -e logs/hyst_recompute_cg_%A_%a.err
#SBATCH --requeue

set -euo pipefail

module load python/3.12.11-fasrc02

cd /n/home06/nchadha/Chiral_DW
source .venv/bin/activate

mkdir -p logs

SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:?SOURCE_OUTPUT_ROOT is required}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${SOURCE_OUTPUT_ROOT%/}_linear_interaction_recomputed"}
MATERIAL=${MATERIAL:-"mote2"}
N_K_LIST=${N_K_LIST:-"18,19,20,21,22,23,24"}
RUN_SLUG=${RUN_SLUG:-"$(basename "${OUTPUT_ROOT%/}")"}
CACHE_BASE_ROOT=${CACHE_BASE_ROOT:-${LAB_SCRATCH_ROOT:-${SCRATCH:-"results/recompute_backend_cache_scratch"}}}
if [[ "$CACHE_BASE_ROOT" != /* ]]; then
  CACHE_BASE_ROOT="$(pwd)/${CACHE_BASE_ROOT}"
fi

U_D_MIN=${U_D_MIN:-"0.0"}
U_D_MAX=${U_D_MAX:-"20.0"}
N_U_D=${N_U_D:-"21"}
THETA_MIN_DEG=${THETA_MIN_DEG:-"2.0"}
THETA_MAX_DEG=${THETA_MAX_DEG:-"4.0"}
N_TWIST=${N_TWIST:-"21"}

PLANE_WAVE_SHELL=${PLANE_WAVE_SHELL:-"5"}
N_BANDS=${N_BANDS:-"2"}
N_ACTIVE_BANDS_PER_VALLEY=${N_ACTIVE_BANDS_PER_VALLEY:-"2"}
Q_MESH=${Q_MESH:-"full"}
Q_SHELL=${Q_SHELL:-"0"}
LOCAL_FIELD_CUTOFF=${LOCAL_FIELD_CUTOFF:-"4"}
INCLUDE_Q0=${INCLUDE_Q0:-"1"}
EPSILON=${EPSILON:-"16.7"}
GATE_DISTANCE_NM=${GATE_DISTANCE_NM:-"30.0"}
if [[ "$MATERIAL" == "wse2" ]]; then
  SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.332"}
else
  SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.347"}
fi
V0=${V0:-"1.0"}
EXCHANGE_SCALE=${EXCHANGE_SCALE:-"1.0"}
HARTREE_SCALE=${HARTREE_SCALE:-"1.0"}
VERTEX_WORKERS=${VERTEX_WORKERS:-"${SLURM_CPUS_PER_TASK:-1}"}
EXCHANGE_WORKERS=${EXCHANGE_WORKERS:-"${SLURM_CPUS_PER_TASK:-1}"}
DENSITY_VERTEX_RETENTION=${DENSITY_VERTEX_RETENTION:-"hartree_only"}
DENSITY_VERTEX_LAYOUT=${DENSITY_VERTEX_LAYOUT:-"auto"}
EXCHANGE_REPRESENTATION=${EXCHANGE_REPRESENTATION:-"auto"}
FORM_FACTOR_BACKEND=${FORM_FACTOR_BACKEND:-"auto"}

N_OCC_PER_K=${N_OCC_PER_K:-"1"}
MAX_ITER=${MAX_ITER:-"800"}
MIN_ITER=${MIN_ITER:-"3"}
MIXING_METHOD=${MIXING_METHOD:-"oda"}
MIXING=${MIXING:-"0.45"}
TOLERANCE=${TOLERANCE:-"1e-8"}
ENERGY_TOLERANCE=${ENERGY_TOLERANCE:-"1e-10"}
FINAL_RESIDUAL_TOLERANCE=${FINAL_RESIDUAL_TOLERANCE:-"1e-7"}

N_THETA=${N_THETA:-"81"}
TRIAL_INTERPOLATION=${TRIAL_INTERPOLATION:-"linear_interaction"}
ENDPOINT_EPS=${ENDPOINT_EPS:-"1e-5"}
DOMAIN_RADIUS=${DOMAIN_RADIUS:-"20.0"}
DOMAIN_WIDTH=${DOMAIN_WIDTH:-"3.0"}
DOMAIN_WINDING=${DOMAIN_WINDING:-"1"}
NAN_TEXTURE_WHEN_IVC_LOWER=${NAN_TEXTURE_WHEN_IVC_LOWER:-"1"}
TEXTURE_ENERGY_TIE_ATOL=${TEXTURE_ENERGY_TIE_ATOL:-"1e-9"}
COMPUTE_INVALID_TEXTURE_CG=${COMPUTE_INVALID_TEXTURE_CG:-"0"}
RERUN_EXISTING=${RERUN_EXISTING:-"0"}
REQUIRE_CACHE=${REQUIRE_CACHE:-"0"}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

IFS=',' read -r -a nks <<< "$N_K_LIST"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID >= ${#nks[@]} )); then
  echo "Task ${TASK_ID} is outside recompute mesh count ${#nks[@]}; exiting."
  exit 0
fi
N_K="$(echo "${nks[$TASK_ID]}" | xargs)"
nk_dir="$(printf "nk_%03d" "$N_K")"

if [[ -f "${SOURCE_OUTPUT_ROOT%/}/hysteresis_all_branch_candidates.csv" && "${#nks[@]}" -eq 1 ]]; then
  source_mesh_root="${SOURCE_OUTPUT_ROOT%/}"
  output_mesh_root="${OUTPUT_ROOT%/}"
else
  source_mesh_root="${SOURCE_OUTPUT_ROOT%/}/${nk_dir}"
  output_mesh_root="${OUTPUT_ROOT%/}/${nk_dir}"
fi

if [[ -n "${CACHE_ROOT:-}" ]]; then
  if [[ "$(basename "${CACHE_ROOT%/}")" == "backend_cache" ]]; then
    cache_root="${CACHE_ROOT%/}"
  else
    cache_root="${CACHE_ROOT%/}/${nk_dir}/backend_cache"
  fi
else
  cache_root="${CACHE_BASE_ROOT%/}/${RUN_SLUG}/${nk_dir}/backend_cache"
fi

OMIT_Q0_FLAG=()
if [[ "$INCLUDE_Q0" == "0" ]]; then
  OMIT_Q0_FLAG=(--omit-q0)
fi
TEXTURE_FLAG=()
if [[ "$NAN_TEXTURE_WHEN_IVC_LOWER" == "0" ]]; then
  TEXTURE_FLAG=(--allow-texture-in-ivc-ground-state)
fi
DIAGNOSTIC_CG_FLAG=()
if [[ "$COMPUTE_INVALID_TEXTURE_CG" == "1" ]]; then
  DIAGNOSTIC_CG_FLAG=(--compute-invalid-texture-cg)
fi
RERUN_FLAG=()
if [[ "$RERUN_EXISTING" == "1" ]]; then
  RERUN_FLAG=(--rerun-existing)
fi
REQUIRE_CACHE_FLAG=()
if [[ "$REQUIRE_CACHE" == "1" ]]; then
  REQUIRE_CACHE_FLAG=(--require-cache)
fi

echo "Recomputing ${MATERIAL} hysteresis cG for n_k=${N_K}"
echo "Source=${source_mesh_root} Output=${output_mesh_root} Cache=${cache_root}"
echo "Resources: SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset} VERTEX_WORKERS=${VERTEX_WORKERS} EXCHANGE_WORKERS=${EXCHANGE_WORKERS} TRIAL_INTERPOLATION=${TRIAL_INTERPOLATION} SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset} SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-unset}"

python scripts/recompute_hysteresis_cg_from_projectors.py \
  --material "$MATERIAL" \
  --source-output-root "$source_mesh_root" \
  --output-root "$output_mesh_root" \
  --cache-root "$cache_root" \
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
  --n-theta "$N_THETA" \
  --trial-interpolation "$TRIAL_INTERPOLATION" \
  --endpoint-eps "$ENDPOINT_EPS" \
  --domain-radius "$DOMAIN_RADIUS" \
  --domain-width "$DOMAIN_WIDTH" \
  --domain-winding "$DOMAIN_WINDING" \
  "${TEXTURE_FLAG[@]}" \
  --texture-energy-tie-atol "$TEXTURE_ENERGY_TIE_ATOL" \
  "${DIAGNOSTIC_CG_FLAG[@]}" \
  "${RERUN_FLAG[@]}" \
  "${REQUIRE_CACHE_FLAG[@]}"
