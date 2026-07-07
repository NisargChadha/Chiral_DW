#!/bin/bash
# Submit recomputation of linear-interaction cG from stored hysteresis projectors.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT:?SOURCE_OUTPUT_ROOT is required}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${SOURCE_OUTPUT_ROOT%/}_linear_interaction_recomputed"}
MATERIAL=${MATERIAL:-"mote2"}
N_K_LIST=${N_K_LIST:-"18,19,20,21,22,23,24"}
FINAL_N_K_LIST=${FINAL_N_K_LIST:-"$N_K_LIST"}
NK_MEMORY_GB_MAP=${NK_MEMORY_GB_MAP:-"18:12,19:14,20:16,21:18,22:20,23:22,24:24"}
RUN_SLUG=${RUN_SLUG:-"$(basename "${OUTPUT_ROOT%/}")"}
CACHE_BASE_ROOT=${CACHE_BASE_ROOT:-${LAB_SCRATCH_ROOT:-${SCRATCH:-"results/recompute_backend_cache_scratch"}}}
if [[ "$CACHE_BASE_ROOT" != /* ]]; then
  CACHE_BASE_ROOT="${ROOT}/${CACHE_BASE_ROOT}"
fi

U_D_MIN=${U_D_MIN:-"0.0"}
U_D_MAX=${U_D_MAX:-"20.0"}
N_U_D=${N_U_D:-"21"}
THETA_MIN_DEG=${THETA_MIN_DEG:-"2.0"}
THETA_MAX_DEG=${THETA_MAX_DEG:-"4.0"}
N_TWIST=${N_TWIST:-"21"}

CPUS_PER_TASK=${CPUS_PER_TASK:-"4"}
VERTEX_WORKERS=${VERTEX_WORKERS:-"$CPUS_PER_TASK"}
EXCHANGE_WORKERS=${EXCHANGE_WORKERS:-"$CPUS_PER_TASK"}
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
DENSITY_VERTEX_RETENTION=${DENSITY_VERTEX_RETENTION:-"hartree_only"}
DENSITY_VERTEX_LAYOUT=${DENSITY_VERTEX_LAYOUT:-"auto"}
EXCHANGE_REPRESENTATION=${EXCHANGE_REPRESENTATION:-"auto"}
FORM_FACTOR_BACKEND=${FORM_FACTOR_BACKEND:-"auto"}

MAX_ITER=${MAX_ITER:-"800"}
MIN_ITER=${MIN_ITER:-"3"}
MIXING_METHOD=${MIXING_METHOD:-"oda"}
MIXING=${MIXING:-"0.45"}
TOLERANCE=${TOLERANCE:-"1e-8"}
ENERGY_TOLERANCE=${ENERGY_TOLERANCE:-"1e-10"}
FINAL_RESIDUAL_TOLERANCE=${FINAL_RESIDUAL_TOLERANCE:-"1e-7"}
N_THETA=${N_THETA:-"81"}
TRIAL_INTERPOLATION=${TRIAL_INTERPOLATION:-"linear_interaction"}
N_OCC_PER_K=${N_OCC_PER_K:-"1"}
COMPUTE_INVALID_TEXTURE_CG=${COMPUTE_INVALID_TEXTURE_CG:-"0"}
NAN_TEXTURE_WHEN_IVC_LOWER=${NAN_TEXTURE_WHEN_IVC_LOWER:-"1"}

RECOMPUTE_TIME=${RECOMPUTE_TIME:-"24:00:00"}
FINAL_MERGE_TIME=${FINAL_MERGE_TIME:-"02:00:00"}
FINAL_MERGE_MEM_GB=${FINAL_MERGE_MEM_GB:-"8"}
SUBMIT_FINAL_MERGE=${SUBMIT_FINAL_MERGE:-"1"}
SEQUENTIAL_MESHES=${SEQUENTIAL_MESHES:-"0"}
DEPENDENCY_AFTEROK=${DEPENDENCY_AFTEROK:-""}
FINAL_DEPENDENCY_AFTEROK=${FINAL_DEPENDENCY_AFTEROK:-""}
DRY_RUN=${DRY_RUN:-"0"}
RERUN_EXISTING=${RERUN_EXISTING:-"0"}
REQUIRE_CACHE=${REQUIRE_CACHE:-"0"}

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$CACHE_BASE_ROOT"
fi

quote_command() {
  printf "%q " "$@"
}

memory_gb_for_nk() {
  local nk="$1"
  local entry key value
  IFS=',' read -r -a entries <<< "$NK_MEMORY_GB_MAP"
  for entry in "${entries[@]}"; do
    key="${entry%%:*}"
    value="${entry##*:}"
    key="$(echo "$key" | xargs)"
    value="$(echo "$value" | xargs)"
    if [[ "$key" == "$nk" ]]; then
      printf "%s" "$value"
      return 0
    fi
  done
  echo "No memory entry for n_k=${nk} in NK_MEMORY_GB_MAP=${NK_MEMORY_GB_MAP}" >&2
  return 1
}

submit_or_echo() {
  if [[ "$DRY_RUN" == "1" ]]; then
    quote_command "$@"
    printf "\n"
    return 0
  fi
  "$@"
}

barrier_job_ids=()
previous_job=""
mesh_count=0
IFS=',' read -r -a nks <<< "$N_K_LIST"
for raw_nk in "${nks[@]}"; do
  nk="$(echo "$raw_nk" | xargs)"
  [[ -z "$nk" ]] && continue
  mesh_count=$((mesh_count + 1))
  mem_gb="$(memory_gb_for_nk "$nk")"
  nk_dir="$(printf "nk_%03d" "$nk")"
  cache_root="${CACHE_BASE_ROOT}/${RUN_SLUG}/${nk_dir}/backend_cache"
  mesh_env=(
    env
    "SOURCE_OUTPUT_ROOT=${SOURCE_OUTPUT_ROOT}"
    "OUTPUT_ROOT=${OUTPUT_ROOT}"
    "MATERIAL=${MATERIAL}"
    "N_K_LIST=${nk}"
    "CACHE_ROOT=${cache_root}"
    "CACHE_BASE_ROOT=${CACHE_BASE_ROOT}"
    "RUN_SLUG=${RUN_SLUG}"
    "U_D_MIN=${U_D_MIN}"
    "U_D_MAX=${U_D_MAX}"
    "N_U_D=${N_U_D}"
    "THETA_MIN_DEG=${THETA_MIN_DEG}"
    "THETA_MAX_DEG=${THETA_MAX_DEG}"
    "N_TWIST=${N_TWIST}"
    "VERTEX_WORKERS=${VERTEX_WORKERS}"
    "EXCHANGE_WORKERS=${EXCHANGE_WORKERS}"
    "PLANE_WAVE_SHELL=${PLANE_WAVE_SHELL}"
    "N_BANDS=${N_BANDS}"
    "N_ACTIVE_BANDS_PER_VALLEY=${N_ACTIVE_BANDS_PER_VALLEY}"
    "Q_MESH=${Q_MESH}"
    "Q_SHELL=${Q_SHELL}"
    "LOCAL_FIELD_CUTOFF=${LOCAL_FIELD_CUTOFF}"
    "INCLUDE_Q0=${INCLUDE_Q0}"
    "EPSILON=${EPSILON}"
    "GATE_DISTANCE_NM=${GATE_DISTANCE_NM}"
    "SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM}"
    "V0=${V0}"
    "EXCHANGE_SCALE=${EXCHANGE_SCALE}"
    "HARTREE_SCALE=${HARTREE_SCALE}"
    "DENSITY_VERTEX_RETENTION=${DENSITY_VERTEX_RETENTION}"
    "DENSITY_VERTEX_LAYOUT=${DENSITY_VERTEX_LAYOUT}"
    "EXCHANGE_REPRESENTATION=${EXCHANGE_REPRESENTATION}"
    "FORM_FACTOR_BACKEND=${FORM_FACTOR_BACKEND}"
    "MAX_ITER=${MAX_ITER}"
    "MIN_ITER=${MIN_ITER}"
    "MIXING_METHOD=${MIXING_METHOD}"
    "MIXING=${MIXING}"
    "TOLERANCE=${TOLERANCE}"
    "ENERGY_TOLERANCE=${ENERGY_TOLERANCE}"
    "FINAL_RESIDUAL_TOLERANCE=${FINAL_RESIDUAL_TOLERANCE}"
    "N_THETA=${N_THETA}"
    "TRIAL_INTERPOLATION=${TRIAL_INTERPOLATION}"
    "N_OCC_PER_K=${N_OCC_PER_K}"
    "COMPUTE_INVALID_TEXTURE_CG=${COMPUTE_INVALID_TEXTURE_CG}"
    "NAN_TEXTURE_WHEN_IVC_LOWER=${NAN_TEXTURE_WHEN_IVC_LOWER}"
    "RERUN_EXISTING=${RERUN_EXISTING}"
    "REQUIRE_CACHE=${REQUIRE_CACHE}"
  )

  recompute_cmd=(
    "${mesh_env[@]}"
    sbatch --parsable
    "--array=0-0"
    "--mem=${mem_gb}G"
    "--time=${RECOMPUTE_TIME}"
    "--cpus-per-task=${CPUS_PER_TASK}"
  )
  dependencies=()
  if [[ "$SEQUENTIAL_MESHES" == "1" && -n "$previous_job" ]]; then
    dependencies+=("$previous_job")
  elif [[ -n "$DEPENDENCY_AFTEROK" ]]; then
    dependencies+=("$DEPENDENCY_AFTEROK")
  fi
  if [[ "${#dependencies[@]}" -gt 0 ]]; then
    IFS=:
    recompute_cmd+=(--dependency=afterok:"${dependencies[*]}")
    unset IFS
  fi
  recompute_cmd+=(
    "--export=ALL"
    jobs/recompute_hysteresis_cg_from_projectors_by_mesh_array.sh
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    submit_or_echo "${recompute_cmd[@]}"
    recompute_job="dry_recompute_${nk}"
  else
    recompute_job="$(submit_or_echo "${recompute_cmd[@]}")"
  fi
  previous_job="$recompute_job"
  barrier_job_ids+=("$recompute_job")
  if [[ "$DRY_RUN" != "1" ]]; then
    echo "n_k=${nk} recompute=${recompute_job} cache_root=${cache_root}"
  fi
done

if [[ "$MATERIAL" == "wse2" ]]; then
  final_merge_job_script="jobs/merge_wse2_ivc_hysteresis_finite_size.sh"
else
  final_merge_job_script="jobs/merge_taige_ivc_hysteresis_finite_size.sh"
fi

final_job="disabled"
if [[ "$SUBMIT_FINAL_MERGE" == "1" ]]; then
  final_dependencies=()
  final_cmd=(
    env
    "OUTPUT_ROOT=${OUTPUT_ROOT}"
    "N_K_LIST=${FINAL_N_K_LIST}"
    "MESH_DIR_TEMPLATE=nk_{n_k:03d}"
    "TRIAL_INTERPOLATION=${TRIAL_INTERPOLATION}"
    sbatch --parsable
    "--mem=${FINAL_MERGE_MEM_GB}G"
    "--time=${FINAL_MERGE_TIME}"
    "--cpus-per-task=1"
  )
  if [[ "${#barrier_job_ids[@]}" -gt 0 ]]; then
    final_dependencies+=("${barrier_job_ids[@]}")
  fi
  if [[ -n "$FINAL_DEPENDENCY_AFTEROK" ]]; then
    final_dependencies+=("$FINAL_DEPENDENCY_AFTEROK")
  fi
  if [[ "${#final_dependencies[@]}" -gt 0 ]]; then
    IFS=:
    final_cmd+=(--dependency=afterok:"${final_dependencies[*]}")
    unset IFS
  fi
  final_cmd+=(
    "--export=ALL"
    "$final_merge_job_script"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    submit_or_echo "${final_cmd[@]}"
    final_job="dry_final_merge"
  else
    final_job="$(submit_or_echo "${final_cmd[@]}")"
  fi
fi

if [[ "$DRY_RUN" != "1" ]]; then
  echo "finite_size_merge=${final_job}"
else
  echo "Scratch cache base: ${CACHE_BASE_ROOT}"
  echo "Dry run task counts: recompute=${mesh_count} final_merge=${SUBMIT_FINAL_MERGE}"
fi
