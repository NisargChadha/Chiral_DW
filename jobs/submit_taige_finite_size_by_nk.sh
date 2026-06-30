#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT=${OUTPUT_ROOT:-"results/taige_cg_finite_size_nk18_24_u0_15_theta2_4p2"}
N_K_LIST=${N_K_LIST:-"18,20,22,24"}
U_D_MIN=${U_D_MIN:-"0.0"}
U_D_MAX=${U_D_MAX:-"15.0"}
N_U_D=${N_U_D:-"21"}
THETA_MIN_DEG=${THETA_MIN_DEG:-"2.0"}
THETA_MAX_DEG=${THETA_MAX_DEG:-"4.2"}
N_TWIST=${N_TWIST:-"21"}

CPUS_PER_TASK=${CPUS_PER_TASK:-"4"}
VERTEX_WORKERS=${VERTEX_WORKERS:-"$CPUS_PER_TASK"}
EXCHANGE_WORKERS=${EXCHANGE_WORKERS:-"$CPUS_PER_TASK"}
EXCHANGE_REPRESENTATION=${EXCHANGE_REPRESENTATION:-"auto"}
FORM_FACTOR_BACKEND=${FORM_FACTOR_BACKEND:-"auto"}
MEMORY_GB_PER_NK=${MEMORY_GB_PER_NK:-"1"}
MEMORY_GB_MIN=${MEMORY_GB_MIN:-"8"}
MAX_CONCURRENT_PER_NK=${MAX_CONCURRENT_PER_NK:-""}
DRY_RUN=${DRY_RUN:-"0"}
MANIFEST_PATH=${MANIFEST_PATH:-"${OUTPUT_ROOT}/slurm_jobs_finite_size_by_nk.csv"}

TOTAL_TASKS=$((N_U_D * N_TWIST))
ARRAY_SPEC="0-$((TOTAL_TASKS - 1))"
if [[ -n "$MAX_CONCURRENT_PER_NK" ]]; then
  ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT_PER_NK}"
fi

memory_gb_for_nk() {
  local nk="$1"
  awk -v nk="$nk" -v per="$MEMORY_GB_PER_NK" -v min="$MEMORY_GB_MIN" 'BEGIN {
    mem = nk * per
    if (mem < min) {
      mem = min
    }
    printf "%d", (mem == int(mem) ? mem : int(mem) + 1)
  }'
}

quote_command() {
  printf "%q " "$@"
}

csv_escape() {
  local value="${1//\"/\"\"}"
  printf '"%s"' "$value"
}

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$(dirname "$MANIFEST_PATH")"
  echo "job_id,n_k,mem_gb,cpus_per_task,array_spec,output_root,submitted_at,command" > "$MANIFEST_PATH"
fi

IFS=',' read -r -a NKS <<< "$N_K_LIST"
for raw_nk in "${NKS[@]}"; do
  nk="$(echo "$raw_nk" | xargs)"
  if [[ -z "$nk" ]]; then
    continue
  fi
  mem_gb="$(memory_gb_for_nk "$nk")"
  export_arg="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},N_K_LIST=${nk},U_D_MIN=${U_D_MIN},U_D_MAX=${U_D_MAX},N_U_D=${N_U_D},THETA_MIN_DEG=${THETA_MIN_DEG},THETA_MAX_DEG=${THETA_MAX_DEG},N_TWIST=${N_TWIST},VERTEX_WORKERS=${VERTEX_WORKERS},EXCHANGE_WORKERS=${EXCHANGE_WORKERS},EXCHANGE_REPRESENTATION=${EXCHANGE_REPRESENTATION},FORM_FACTOR_BACKEND=${FORM_FACTOR_BACKEND}"
  cmd=(
    sbatch
    "--array=${ARRAY_SPEC}"
    "--mem=${mem_gb}G"
    "--cpus-per-task=${CPUS_PER_TASK}"
    "--export=${export_arg}"
    jobs/scan_taige_finite_size_cg_array.sh
  )
  command_text="$(quote_command "${cmd[@]}")"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "$command_text"
    continue
  fi
  output="$("${cmd[@]}")"
  echo "$output"
  job_id="$(awk '/Submitted batch job/ {print $4}' <<< "$output")"
  submitted_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  {
    csv_escape "$job_id"; printf ","
    csv_escape "$nk"; printf ","
    csv_escape "$mem_gb"; printf ","
    csv_escape "$CPUS_PER_TASK"; printf ","
    csv_escape "$ARRAY_SPEC"; printf ","
    csv_escape "$OUTPUT_ROOT"; printf ","
    csv_escape "$submitted_at"; printf ","
    csv_escape "$command_text"; printf "\n"
  } >> "$MANIFEST_PATH"
done

if [[ "$DRY_RUN" != "1" ]]; then
  echo "Wrote SLURM manifest to ${MANIFEST_PATH}"
fi
