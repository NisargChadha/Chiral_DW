#!/bin/bash
# Submit WSe2 IVC hysteresis finite-size pipeline with mesh-specific memory.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT=${OUTPUT_ROOT:-"results/wse2_ivc_hysteresis_linear_interaction_finite_size_nk18_22_grid41"}
N_K_LIST=${N_K_LIST:-"18,19,20,21,22"}
FINAL_N_K_LIST=${FINAL_N_K_LIST:-"$N_K_LIST"}
NK_MEMORY_GB_MAP=${NK_MEMORY_GB_MAP:-"18:12,19:14,20:16,21:18,22:20,23:22,24:24"}
SEQUENTIAL_MESHES=${SEQUENTIAL_MESHES:-"1"}
RUN_SLUG=${RUN_SLUG:-"$(basename "${OUTPUT_ROOT%/}")"}
CACHE_BASE_ROOT=${CACHE_BASE_ROOT:-${LAB_SCRATCH_ROOT:-${SCRATCH:-"results/wse2_backend_cache_scratch"}}}
if [[ "$CACHE_BASE_ROOT" != /* ]]; then
  CACHE_BASE_ROOT="${ROOT}/${CACHE_BASE_ROOT}"
fi

U_D_MIN=${U_D_MIN:-"0.0"}
U_D_MAX=${U_D_MAX:-"20.0"}
N_U_D=${N_U_D:-"41"}
THETA_MIN_DEG=${THETA_MIN_DEG:-"2.0"}
THETA_MAX_DEG=${THETA_MAX_DEG:-"4.0"}
N_TWIST=${N_TWIST:-"41"}

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
SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.332"}
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
RANDOM_SEEDS=${RANDOM_SEEDS:-"1,7,13,29,53"}
N_THETA=${N_THETA:-"81"}
TRIAL_INTERPOLATION=${TRIAL_INTERPOLATION:-"linear_interaction"}
N_OCC_PER_K=${N_OCC_PER_K:-"1"}
SOLVE_VP_REFERENCES=${SOLVE_VP_REFERENCES:-"1"}
COMPUTE_VP_CHERN=${COMPUTE_VP_CHERN:-"1"}
COMPUTE_INVALID_TEXTURE_CG=${COMPUTE_INVALID_TEXTURE_CG:-"0"}

CACHE_TIME=${CACHE_TIME:-"24:00:00"}
SCAN_TIME=${SCAN_TIME:-"24:00:00"}
MERGE_TIME=${MERGE_TIME:-"02:00:00"}
FINAL_MERGE_TIME=${FINAL_MERGE_TIME:-"02:00:00"}
MERGE_MEM_GB=${MERGE_MEM_GB:-"8"}
FINAL_MERGE_MEM_GB=${FINAL_MERGE_MEM_GB:-"8"}
CLEANUP_BACKEND_CACHE=${CLEANUP_BACKEND_CACHE:-"1"}
CLEANUP_TIME=${CLEANUP_TIME:-"01:00:00"}
CLEANUP_MEM_GB=${CLEANUP_MEM_GB:-"2"}
MAX_CONCURRENT_CACHE=${MAX_CONCURRENT_CACHE:-""}
MAX_CONCURRENT_SCAN=${MAX_CONCURRENT_SCAN:-""}
SUBMIT_FINAL_MERGE=${SUBMIT_FINAL_MERGE:-"1"}
DEPENDENCY_AFTEROK=${DEPENDENCY_AFTEROK:-""}
FINAL_DEPENDENCY_AFTEROK=${FINAL_DEPENDENCY_AFTEROK:-""}
DRY_RUN=${DRY_RUN:-"0"}

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$CACHE_BASE_ROOT"
fi

total_cache_tasks=$((N_U_D * N_TWIST))
total_scan_tasks=$((2 * (N_U_D + N_TWIST)))
cache_array="0-$((total_cache_tasks - 1))"
scan_array="0-$((total_scan_tasks - 1))"
if [[ -n "$MAX_CONCURRENT_CACHE" ]]; then
  cache_array="${cache_array}%${MAX_CONCURRENT_CACHE}"
fi
if [[ -n "$MAX_CONCURRENT_SCAN" ]]; then
  scan_array="${scan_array}%${MAX_CONCURRENT_SCAN}"
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
previous_barrier_job=""
mesh_count=0
IFS=',' read -r -a nks <<< "$N_K_LIST"
for raw_nk in "${nks[@]}"; do
  nk="$(echo "$raw_nk" | xargs)"
  [[ -z "$nk" ]] && continue
  mesh_count=$((mesh_count + 1))
  mem_gb="$(memory_gb_for_nk "$nk")"
  nk_dir="$(printf "nk_%03d" "$nk")"
  mesh_root="${OUTPUT_ROOT}/${nk_dir}"
  cache_root="${CACHE_BASE_ROOT}/${RUN_SLUG}/${nk_dir}/backend_cache"
  mesh_env=(
    env
    "OUTPUT_ROOT=${mesh_root}"
    "CACHE_ROOT=${cache_root}"
    "CACHE_BASE_ROOT=${CACHE_BASE_ROOT}"
    "RUN_SLUG=${RUN_SLUG}"
    "N_K=${nk}"
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
    "RANDOM_SEEDS=${RANDOM_SEEDS}"
    "N_THETA=${N_THETA}"
    "TRIAL_INTERPOLATION=${TRIAL_INTERPOLATION}"
    "N_OCC_PER_K=${N_OCC_PER_K}"
    "SOLVE_VP_REFERENCES=${SOLVE_VP_REFERENCES}"
    "COMPUTE_VP_CHERN=${COMPUTE_VP_CHERN}"
    "COMPUTE_INVALID_TEXTURE_CG=${COMPUTE_INVALID_TEXTURE_CG}"
    "CLEANUP_BACKEND_CACHE=${CLEANUP_BACKEND_CACHE}"
  )

  cache_cmd=(
    "${mesh_env[@]}"
    sbatch --parsable
    "--array=${cache_array}"
    "--mem=${mem_gb}G"
    "--time=${CACHE_TIME}"
    "--cpus-per-task=${CPUS_PER_TASK}"
  )
  if [[ "$SEQUENTIAL_MESHES" == "1" && -n "$previous_barrier_job" ]]; then
    cache_cmd+=(--dependency=afterok:"$previous_barrier_job")
  elif [[ -n "$DEPENDENCY_AFTEROK" ]]; then
    cache_cmd+=(--dependency=afterok:"$DEPENDENCY_AFTEROK")
  fi
  cache_cmd+=(
    "--export=ALL"
    jobs/precompute_wse2_backend_cache_array.sh
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    submit_or_echo "${cache_cmd[@]}"
    cache_job="dry_cache_${nk}"
  else
    cache_job="$(submit_or_echo "${cache_cmd[@]}")"
  fi

  scan_cmd=(
    "${mesh_env[@]}"
    sbatch --parsable
    "--array=${scan_array}"
    "--mem=${mem_gb}G"
    "--time=${SCAN_TIME}"
    "--cpus-per-task=${CPUS_PER_TASK}"
  )
  scan_cmd+=(--dependency=afterok:"$cache_job")
  scan_cmd+=(
    "--export=ALL"
    jobs/scan_wse2_ivc_hysteresis_all_linecuts_array.sh
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    submit_or_echo "${scan_cmd[@]}"
    scan_job="dry_scan_${nk}"
  else
    scan_job="$(submit_or_echo "${scan_cmd[@]}")"
  fi

  merge_cmd=(
    "${mesh_env[@]}"
    sbatch --parsable
    "--mem=${MERGE_MEM_GB}G"
    "--time=${MERGE_TIME}"
    "--cpus-per-task=1"
  )
  merge_cmd+=(--dependency=afterok:"$scan_job")
  merge_cmd+=(
    "--export=ALL"
    jobs/merge_wse2_ivc_hysteresis_sweep.sh
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    submit_or_echo "${merge_cmd[@]}"
    merge_job="dry_merge_${nk}"
  else
    merge_job="$(submit_or_echo "${merge_cmd[@]}")"
  fi
  barrier_job="$merge_job"
  cleanup_job=""
  if [[ "$CLEANUP_BACKEND_CACHE" == "1" ]]; then
    cleanup_cmd=(
      "${mesh_env[@]}"
      sbatch --parsable
      "--mem=${CLEANUP_MEM_GB}G"
      "--time=${CLEANUP_TIME}"
      "--cpus-per-task=1"
      --dependency=afterok:"$merge_job"
      "--export=ALL"
      jobs/cleanup_wse2_backend_cache.sh
    )
    if [[ "$DRY_RUN" == "1" ]]; then
      submit_or_echo "${cleanup_cmd[@]}"
      cleanup_job="dry_cleanup_${nk}"
    else
      cleanup_job="$(submit_or_echo "${cleanup_cmd[@]}")"
    fi
    barrier_job="$cleanup_job"
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    barrier_job_ids+=("$barrier_job")
    echo "n_k=${nk} cache=${cache_job} scan=${scan_job} merge=${merge_job} cleanup=${cleanup_job:-disabled} cache_root=${cache_root}"
  fi
  previous_barrier_job="$barrier_job"
done

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
  if [[ "$DRY_RUN" != "1" && "${#barrier_job_ids[@]}" -gt 0 ]]; then
    final_dependencies+=("${barrier_job_ids[@]}")
  elif [[ "$DRY_RUN" == "1" && "$SEQUENTIAL_MESHES" == "1" && -n "$previous_barrier_job" ]]; then
    final_dependencies+=("$previous_barrier_job")
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
    jobs/merge_wse2_ivc_hysteresis_finite_size.sh
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
  echo "Dry run task counts: cache=$((total_cache_tasks * mesh_count)) scan=$((total_scan_tasks * mesh_count)) final_merge=${SUBMIT_FINAL_MERGE}"
fi
