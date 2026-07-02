#!/bin/bash
# Submit the full Taige IVC hysteresis pipeline with one command.

set -euo pipefail

CACHE_JOB=${CACHE_JOB:-"jobs/precompute_taige_backend_cache_array.sh"}
LINECUT_JOB=${LINECUT_JOB:-"jobs/scan_taige_ivc_hysteresis_all_linecuts_array.sh"}
MERGE_JOB=${MERGE_JOB:-"jobs/merge_taige_ivc_hysteresis_sweep.sh"}

CACHE_JOB_ID=$(sbatch --parsable "$CACHE_JOB")
LINECUT_JOB_ID=$(sbatch --parsable --dependency=afterok:"$CACHE_JOB_ID" "$LINECUT_JOB")
MERGE_JOB_ID=$(sbatch --parsable --dependency=afterok:"$LINECUT_JOB_ID" "$MERGE_JOB")

echo "Submitted Taige IVC hysteresis pipeline"
echo "  cache:   ${CACHE_JOB_ID}"
echo "  linecut: ${LINECUT_JOB_ID}"
echo "  merge:   ${MERGE_JOB_ID}"
