#!/usr/bin/env bash
set -euo pipefail

GPU_RANDOM="cuda:0"
GPU_BUTINA="cuda:1"
NUM_EPOCHS=200
PATIENCE=5

LOG_DIR="logs/logs_gnn_10endpoints"
mkdir -p "${LOG_DIR}"

RANDOM_PARAMS="best_hyperparameters_random_10endpoints.json"
BUTINA_PARAMS="best_hyperparameters_butina_10endpoints.json"

run_and_log() {
    local name="$1"
    local logfile="$2"
    shift 2

    echo "[$(date '+%F %T')] START ${name}"
    echo "[$(date '+%F %T')] Command: $*" > "${logfile}"
    "$@" >> "${logfile}" 2>&1
    echo "[$(date '+%F %T')] DONE ${name}"
}

if [[ ! -f "${RANDOM_PARAMS}" ]]; then
    echo "[$(date '+%F %T')] ERROR: hyperparameter file not found: ${RANDOM_PARAMS}" >&2
    exit 1
fi
if [[ ! -f "${BUTINA_PARAMS}" ]]; then
    echo "[$(date '+%F %T')] ERROR: hyperparameter file not found: ${BUTINA_PARAMS}" >&2
    exit 1
fi

echo "[$(date '+%F %T')] Starting GNN training on two GPUs"

run_and_log "train-random-seed2878" "${LOG_DIR}/gnn_train_random_seed2878.log" \
    python -u gnn_train.py \
        --csv-path "./NR_10_endpoints/random/seed_2878/raw/NR_with_split_seed2878.csv" \
        --dataset-root "./NR_10_endpoints/random/seed_2878" \
        --params-path "${RANDOM_PARAMS}" \
        --save-dir "./gnn_results_10endpoints/random/seed_2878" \
        --device "${GPU_RANDOM}" \
        --num-epochs "${NUM_EPOCHS}" \
        --patience "${PATIENCE}" &
PID_RANDOM_TRAIN=$!

run_and_log "train-butina-seed3821" "${LOG_DIR}/gnn_train_butina_seed3821.log" \
    python -u gnn_train.py \
        --csv-path "./NR_10_endpoints/butina/seed_3821/raw/NR_with_butina_split_seed3821.csv" \
        --dataset-root "./NR_10_endpoints/butina/seed_3821" \
        --params-path "${BUTINA_PARAMS}" \
        --save-dir "./gnn_results_10endpoints/butina/seed_3821" \
        --device "${GPU_BUTINA}" \
        --num-epochs "${NUM_EPOCHS}" \
        --patience "${PATIENCE}" &
PID_BUTINA_TRAIN=$!

wait "${PID_RANDOM_TRAIN}"
wait "${PID_BUTINA_TRAIN}"

echo "[$(date '+%F %T')] Random and Butina GNN training finished"
