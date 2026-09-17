#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="logs/logs_ml_10endpoints"
mkdir -p "${LOG_DIR}"

run_and_log() {
    local name="$1"
    local logfile="$2"
    shift 2

    echo "[$(date '+%F %T')] START ${name}"
    echo "[$(date '+%F %T')] Command: $*" > "${logfile}"
    "$@" >> "${logfile}" 2>&1
    echo "[$(date '+%F %T')] DONE ${name}"
}

run_split() {
    local split_name="$1"
    local seed="$2"
    local csv_path="$3"
    local output_dir="./ml_results_10endpoints/${split_name}/seed_${seed}"
    local params_path="${output_dir}/params/all_best_hyperparameters.json"

    if [[ ! -f "${csv_path}" ]]; then
        echo "[$(date '+%F %T')] ERROR: split CSV not found: ${csv_path}" >&2
        exit 1
    fi
    if [[ ! -f "${params_path}" ]]; then
        echo "[$(date '+%F %T')] ERROR: hyperparameter file not found: ${params_path}" >&2
        exit 1
    fi

    run_and_log "ml-${split_name}-seed${seed}" "${LOG_DIR}/ml_${split_name}_seed${seed}_fixed_params.log" \
        python -u ml_train.py \
            --csv-path "${csv_path}" \
            --output-dir "${output_dir}" \
            --params-in "${params_path}"
}

echo "[$(date '+%F %T')] Starting ML training"
run_split "random" "2878" \
    "./NR_10_endpoints/random/seed_2878/raw/NR_with_split_seed2878.csv"
run_split "butina" "3821" \
    "./NR_10_endpoints/butina/seed_3821/raw/NR_with_butina_split_seed3821.csv"

echo "[$(date '+%F %T')] Random and Butina ML training finished"
