#!/usr/bin/env bash
# Run the complete NR-Predictor pipeline on a Linux x86_64 server.
#
# Example:
#   nohup bash scripts/run_full_pipeline.sh >/dev/null 2>&1 &
#
# Optional overrides:
#   PYTHON_BIN=/path/to/env/bin/python LIGAND_WORKERS=12 \
#   DOCKING_WORKERS=12 CPU_PER_JOB=4 XGB_N_JOBS=12 \
#   nohup bash scripts/run_full_pipeline.sh >/dev/null 2>&1 &

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
LIGAND_WORKERS="${LIGAND_WORKERS:-12}"
DOCKING_WORKERS="${DOCKING_WORKERS:-12}"
CPU_PER_JOB="${CPU_PER_JOB:-4}"
XGB_N_JOBS="${XGB_N_JOBS:-12}"
VINA_EXE="$PROJECT_ROOT/vina/vina_1.2.5_linux_x86_64"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "$LOG_FILE") 2>&1

CURRENT_STEP="initialization"
trap 'status=$?; echo "[$(date "+%F %T")] ERROR: ${CURRENT_STEP} failed (exit code ${status})."; echo "Log: ${LOG_FILE}"; exit "$status"' ERR

echo "[$(date '+%F %T')] Project root: $PROJECT_ROOT"
echo "[$(date '+%F %T')] Log file: $LOG_FILE"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: This pipeline requires Linux because the bundled Vina executable is Linux x86_64."
    exit 1
fi

if [[ ! -f "$VINA_EXE" ]]; then
    echo "ERROR: Missing Vina executable: $VINA_EXE"
    exit 1
fi

chmod +x "$VINA_EXE"
export PYTHONUNBUFFERED=1

CURRENT_STEP="environment preflight"
"$PYTHON_BIN" -c "import joblib, meeko, numpy, pandas, rdkit, sklearn, xgboost; print('Python dependencies: OK')"
"$VINA_EXE" --version

CURRENT_STEP="1/3 ligand-cache preparation"
echo "[$(date '+%F %T')] Step 1/3: preparing ligand cache"
"$PYTHON_BIN" scripts/prepare_ligand.py cache --workers "$LIGAND_WORKERS"

CURRENT_STEP="2/3 docking and feature assembly"
echo "[$(date '+%F %T')] Step 2/3: docking and feature assembly"
"$PYTHON_BIN" scripts/feature.py all \
    --vina_exe "$VINA_EXE" \
    --ligand_workers "$LIGAND_WORKERS" \
    --docking_workers "$DOCKING_WORKERS" \
    --cpu_per_job "$CPU_PER_JOB"

CURRENT_STEP="3/3 model training"
echo "[$(date '+%F %T')] Step 3/3: training XGBoost models"
"$PYTHON_BIN" scripts/train_model.py --xgb-n-jobs "$XGB_N_JOBS"

echo "[$(date '+%F %T')] Pipeline completed successfully."
echo "Results: $PROJECT_ROOT/results"
echo "Log: $LOG_FILE"
