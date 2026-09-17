#!/usr/bin/env bash
set -euo pipefail

# Run this script from the project root after activating the Python environment.
GPU_RANDOM="${GPU_RANDOM:-cuda:0}"
GPU_BUTINA="${GPU_BUTINA:-cuda:1}"
LOG_DIR="logs/logs_ensemble_10endpoints"
mkdir -p "${LOG_DIR}"

# Default: reuse val/test base-prediction caches if they already exist.
FORCE_ARGS=()
if [[ "${FORCE_BASE_PREDICT:-0}" == "1" ]]; then
    FORCE_ARGS=(--force-base-predict)
fi

require_file() {
    local file_path="$1"
    if [[ ! -f "${file_path}" ]]; then
        echo "[$(date '+%F %T')] ERROR: required file not found: ${file_path}" >&2
        exit 1
    fi
}

run_and_log() {
    local name="$1"
    local logfile="$2"
    shift 2

    echo "[$(date '+%F %T')] START ${name}"
    echo "[$(date '+%F %T')] Command: $*" > "${logfile}"
    "$@" >> "${logfile}" 2>&1
    echo "[$(date '+%F %T')] DONE ${name}"
}

RANDOM_CSV="./NR_10_endpoints/random/seed_2878/raw/NR_with_split_seed2878.csv"
BUTINA_CSV="./NR_10_endpoints/butina/seed_3821/raw/NR_with_butina_split_seed3821.csv"
RANDOM_PARAMS="./best_hyperparameters_random_10endpoints.json"
BUTINA_PARAMS="./best_hyperparameters_butina_10endpoints.json"

require_file "${RANDOM_CSV}"
require_file "${BUTINA_CSV}"
require_file "${RANDOM_PARAMS}"
require_file "${BUTINA_PARAMS}"

echo "[$(date '+%F %T')] Starting attention-stacking training on two GPUs"

run_and_log "ensemble-random-seed2878" "${LOG_DIR}/ensemble_random_seed2878.log" \
    python -u ensemble_attention_stacking.py \
        --split random \
        --seed 2878 \
        --device "${GPU_RANDOM}" \
        "${FORCE_ARGS[@]}" &
PID_RANDOM=$!

run_and_log "ensemble-butina-seed3821" "${LOG_DIR}/ensemble_butina_seed3821.log" \
    python -u ensemble_attention_stacking.py \
        --split butina \
        --seed 3821 \
        --device "${GPU_BUTINA}" \
        "${FORCE_ARGS[@]}" &
PID_BUTINA=$!

wait "${PID_RANDOM}"
wait "${PID_BUTINA}"

echo "[$(date '+%F %T')] Random and Butina attention-stacking training finished"
