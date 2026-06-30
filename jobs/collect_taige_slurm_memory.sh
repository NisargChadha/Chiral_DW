#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MANIFEST_PATH=${1:-${MANIFEST_PATH:-""}}
if [[ -z "$MANIFEST_PATH" ]]; then
  echo "Usage: $0 path/to/slurm_jobs_manifest.csv" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest not found: ${MANIFEST_PATH}" >&2
  exit 2
fi

OUTPUT_PATH=${OUTPUT_PATH:-"$(dirname "$MANIFEST_PATH")/slurm_memory_usage.csv"}
job_ids="$(tail -n +2 "$MANIFEST_PATH" | awk -F',' '{gsub(/^"|"$/, "", $1); if ($1 != "") print $1}' | paste -sd, -)"
if [[ -z "$job_ids" ]]; then
  echo "No job IDs found in ${MANIFEST_PATH}" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"
echo "job_id_raw,state,elapsed,alloc_cpus,req_mem,max_rss,max_rss_mb,max_vmsize,max_vmsize_mb" > "$OUTPUT_PATH"
sacct \
  -j "$job_ids" \
  --format=JobIDRaw,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,MaxVMSize \
  --parsable2 \
  --noheader |
awk -F'|' '
function to_mb(value, trimmed, n, unit) {
  trimmed = value
  gsub(/^[[:space:]]+|[[:space:]]+$/, "", trimmed)
  if (trimmed == "" || trimmed == "Unknown") {
    return ""
  }
  n = trimmed + 0
  unit = toupper(substr(trimmed, length(trimmed), 1))
  if (unit == "K") {
    return sprintf("%.6f", n / 1024.0)
  }
  if (unit == "M") {
    return sprintf("%.6f", n)
  }
  if (unit == "G") {
    return sprintf("%.6f", n * 1024.0)
  }
  if (unit == "T") {
    return sprintf("%.6f", n * 1024.0 * 1024.0)
  }
  return sprintf("%.6f", n / 1024.0)
}
{
  print $1 "," $2 "," $3 "," $4 "," $5 "," $6 "," to_mb($6) "," $7 "," to_mb($7)
}' >> "$OUTPUT_PATH"

echo "Wrote SLURM memory usage to ${OUTPUT_PATH}"
