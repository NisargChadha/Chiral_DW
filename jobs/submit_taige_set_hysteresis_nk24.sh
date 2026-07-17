#!/bin/bash
#SBATCH -J taige_set_hyst24
#SBATCH -p serial_requeue
#SBATCH -t 48:00:00
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -o logs/taige_set_hyst24_%A_%a.out
#SBATCH -e logs/taige_set_hyst24_%A_%a.err
#SBATCH --requeue

# Submit or execute a restartable Nk=24 scanning-SET hysteresis pipeline.

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"results/taige_set_nk24_theta3_u0_10_hysteresis_step0p5"}
SEED_ROOT=${SEED_ROOT:-"results/taige_set_nk24_theta3_u0_10_endpoint_seeds"}

U_D_MIN=${U_D_MIN:-"0.0"}
U_D_MAX=${U_D_MAX:-"10.0"}
N_U_D=${N_U_D:-"21"}
THETA_DEG=${THETA_DEG:-"3.0"}
N_K=${N_K:-"24"}

PLANE_WAVE_SHELL=${PLANE_WAVE_SHELL:-"5"}
N_BANDS=${N_BANDS:-"2"}
N_ACTIVE_BANDS_PER_VALLEY=${N_ACTIVE_BANDS_PER_VALLEY:-"2"}
Q_MESH=${Q_MESH:-"full"}
Q_SHELL=${Q_SHELL:-"0"}
LOCAL_FIELD_CUTOFF=${LOCAL_FIELD_CUTOFF:-"4"}
EPSILON=${EPSILON:-"16.7"}
GATE_DISTANCE_NM=${GATE_DISTANCE_NM:-"30.0"}
SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.347"}
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
FILLING_WORKERS=${FILLING_WORKERS:-"3"}
VERTEX_WORKERS=${VERTEX_WORKERS:-"${SLURM_CPUS_PER_TASK:-4}"}
EXCHANGE_WORKERS=${EXCHANGE_WORKERS:-"${SLURM_CPUS_PER_TASK:-4}"}

u_d_label() {
  local value
  printf -v value "%08.4f" "$1"
  value="${value/-/m}"
  value="${value/./p}"
  printf "uD_%s" "$value"
}

UP_LABEL="$(u_d_label "$U_D_MIN")"
DOWN_LABEL="$(u_d_label "$U_D_MAX")"
UP_SEED_DIR="${SEED_ROOT}/up/points/${UP_LABEL}"
DOWN_SEED_DIR="${SEED_ROOT}/down/points/${DOWN_LABEL}"

quote_command() {
  printf "%q " "$@"
  printf "\n"
}

submit_pipeline() {
  cd "$REPO_ROOT"
  mkdir -p logs

  local seed_cmd=(
    sbatch --parsable
    --export=ALL,PIPELINE_STAGE=seed,SEED_TASK_ID=0
    "$SCRIPT_PATH"
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    quote_command "${seed_cmd[@]}"
    quote_command sbatch --parsable --dependency=afterok:SEED_JOB_ID \
      --export=ALL,PIPELINE_STAGE=smoke "$SCRIPT_PATH"
    echo "A successful smoke job submits the down-endpoint seed, branches, and merge."
    return 0
  fi

  local seed_job_id smoke_job_id
  seed_job_id="$("${seed_cmd[@]}")"
  smoke_job_id="$(
    sbatch --parsable --dependency=afterok:"$seed_job_id" \
      --export=ALL,PIPELINE_STAGE=smoke "$SCRIPT_PATH"
  )"

  echo "Submitted staged Nk=${N_K} Taige SET hysteresis smoke pipeline"
  echo "  up endpoint seed (u_D=${U_D_MIN}): ${seed_job_id}"
  echo "  one-point continuation smoke:      ${smoke_job_id}"
  echo "  state storage:                     projectors only"
  echo "  output:                            ${OUTPUT_ROOT}"
  echo "The verified smoke job will submit the remaining endpoint, branches, and merge."
}

submit_downstream() {
  cd "$REPO_ROOT"
  local smoke_job_id=${SLURM_JOB_ID:?downstream submission requires the smoke job id}
  local marker="${OUTPUT_ROOT}/downstream_jobs.json"
  if [[ -f "$marker" ]]; then
    echo "Downstream jobs were already recorded in ${marker}; not submitting duplicates."
    return 0
  fi

  local down_seed_job_id branch_job_id merge_job_id
  down_seed_job_id="$(
    sbatch --parsable --dependency=afterok:"$smoke_job_id" \
      --export=ALL,PIPELINE_STAGE=seed,SEED_TASK_ID=1 "$SCRIPT_PATH"
  )"
  branch_job_id="$(
    sbatch --parsable --array=0-1 --dependency=afterok:"$down_seed_job_id" \
      --export=ALL,PIPELINE_STAGE=branch "$SCRIPT_PATH"
  )"
  merge_job_id="$(
    sbatch --parsable --dependency=afterok:"$branch_job_id" \
      --export=ALL,PIPELINE_STAGE=merge "$SCRIPT_PATH"
  )"

  python - "$marker" "$smoke_job_id" "$down_seed_job_id" "$branch_job_id" \
    "$merge_job_id" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "schema": "taige_set_hysteresis_downstream_jobs_v1",
    "smoke_job_id": sys.argv[2],
    "down_endpoint_seed_job_id": sys.argv[3],
    "branch_array_job_id": sys.argv[4],
    "merge_job_id": sys.argv[5],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True))
PY

  echo "Smoke passed; submitted the remaining Nk=${N_K} pipeline"
  echo "  down endpoint seed (u_D=${U_D_MAX}): ${down_seed_job_id}"
  echo "  up/down continuation array:          ${branch_job_id}"
  echo "  lower-envelope SET merge:            ${merge_job_id}"
}

if [[ -z "${PIPELINE_STAGE:-}" ]]; then
  submit_pipeline
  exit 0
fi

module load python/3.12.11-fasrc02
cd "$REPO_ROOT"
source .venv/bin/activate
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_seed() {
  local task_id
  if [[ -n "${SEED_TASK_ID:-}" ]]; then
    task_id="$SEED_TASK_ID"
  else
    task_id=${SLURM_ARRAY_TASK_ID:?seed stage requires SEED_TASK_ID or an array task}
  fi
  local u_d seed_output
  case "$task_id" in
    0)
      u_d="$U_D_MIN"
      seed_output="${SEED_ROOT}/up"
      ;;
    1)
      u_d="$U_D_MAX"
      seed_output="${SEED_ROOT}/down"
      ;;
    *)
      echo "Seed task ${task_id} is outside 0-1" >&2
      exit 2
      ;;
  esac

  python scripts/scan_taige_set_spectrum.py \
    --output-root "$seed_output" \
    --u-d "$u_d" \
    --theta-deg "$THETA_DEG" \
    --n-k "$N_K" \
    --particle-offset-max 1 \
    --filling-workers "$FILLING_WORKERS" \
    --plane-wave-shell "$PLANE_WAVE_SHELL" \
    --n-bands "$N_BANDS" \
    --n-active-bands-per-valley "$N_ACTIVE_BANDS_PER_VALLEY" \
    --q-mesh "$Q_MESH" \
    --q-shell "$Q_SHELL" \
    --local-field-cutoff "$LOCAL_FIELD_CUTOFF" \
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
    --max-iter "$MAX_ITER" \
    --min-iter "$MIN_ITER" \
    --mixing-method "$MIXING_METHOD" \
    --mixing "$MIXING" \
    --tolerance "$TOLERANCE" \
    --energy-tolerance "$ENERGY_TOLERANCE" \
    --dos-energy-points 101 \
    --projectors-only \
    --skip-existing
}

run_branch() {
  local direction=$1
  local seed_dir=$2
  shift 2
  python scripts/scan_taige_set_hysteresis.py \
    --output-root "$OUTPUT_ROOT" \
    --direction "$direction" \
    --seed-point-dir "$seed_dir" \
    --u-d-min "$U_D_MIN" \
    --u-d-max "$U_D_MAX" \
    --n-u-d "$N_U_D" \
    --max-iter "$MAX_ITER" \
    --filling-workers "$FILLING_WORKERS" \
    --projectors-only \
    --skip-existing \
    "$@"
}

verify_smoke_artifacts() {
  local point_dir="${OUTPUT_ROOT}/branches/up/${UP_LABEL}"
  python - "$point_dir" "$N_K" "$EPSILON" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

point_dir = Path(sys.argv[1])
n_cells = int(sys.argv[2]) ** 2
epsilon = float(sys.argv[3])
payload = json.loads((point_dir / "point_summary.json").read_text())
summary = payload["summary"]
if summary["direction"] != "up" or len(summary["filling_energy_rows"]) != 3:
    raise SystemExit("invalid SET hysteresis smoke summary")
if payload.get("state_storage", {}).get("mode") != "projectors_only":
    raise SystemExit("smoke point did not use projector-only storage")
params = payload["params"]
if int(params["grid"]["n_k"]) ** 2 != n_cells:
    raise SystemExit("smoke point has the wrong momentum grid")
if not np.isclose(float(params["interaction"]["epsilon"]), epsilon):
    raise SystemExit("smoke point has the wrong dielectric constant")
with np.load(point_dir / "hf_projectors.npz") as archive:
    expected = {f"global_N{n_cells + offset}_P" for offset in (-1, 0, 1)}
    present = set(archive.files)
if present != expected:
    raise SystemExit(
        f"smoke projector archive mismatch: missing={sorted(expected - present)}, "
        f"unexpected={sorted(present - expected)}"
    )
print(f"Verified SET hysteresis smoke artifacts in {point_dir}")
PY
}

case "$PIPELINE_STAGE" in
  seed)
    run_seed
    ;;
  smoke)
    run_branch up "$UP_SEED_DIR" --max-points 1
    verify_smoke_artifacts
    submit_downstream
    ;;
  branch)
    case "${SLURM_ARRAY_TASK_ID:?branch stage requires a two-element array}" in
      0) run_branch up "$UP_SEED_DIR" ;;
      1) run_branch down "$DOWN_SEED_DIR" ;;
      *) echo "Branch task is outside 0-1" >&2; exit 2 ;;
    esac
    ;;
  merge)
    python scripts/scan_taige_set_hysteresis.py \
      --output-root "$OUTPUT_ROOT" \
      --direction merge \
      --u-d-min "$U_D_MIN" \
      --u-d-max "$U_D_MAX" \
      --n-u-d "$N_U_D"
    ;;
  *)
    echo "Unknown PIPELINE_STAGE=${PIPELINE_STAGE}" >&2
    exit 2
    ;;
esac
