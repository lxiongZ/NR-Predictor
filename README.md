# NR-Predictor

NR-Predictor: A Hierarchical Framework for Predicting Ligand Binding and Functional Activity of Nuclear Receptors

<p align="center">
  <img src="Picture/2.png" alt="NR-Predictor logo" width="900">
</p>

The web server can be used directly at: https://lmmd.ecust.edu.cn/nrpre/

## Framework

<p align="center">
  <img src="Picture/1.png" alt="NR-Predictor framework" width="900">
</p>

## Environment Setup

The recommended environment uses Python 3.10.

```bash
conda create -n nrpre python=3.10
conda activate nrpre

pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.6.1
pip install scikit-learn
pip install texttable
conda install -c conda-forge rdkit=2024.3.6
pip install numpy==1.24.4

curl -L -o dgl-1.1.2+cu118-cp310-cp310-manylinux1_x86_64.whl https://data.dgl.ai/wheels/cu118/dgl-1.1.2%2Bcu118-cp310-cp310-manylinux1_x86_64.whl
pip install dgl-1.1.2+cu118-cp310-cp310-manylinux1_x86_64.whl
pip install dgllife

conda install -c conda-forge meeko=0.6.1
pip install xgboost
```

The DGL wheel above is for Linux, Python 3.10, CUDA 11.8. If your CUDA, Python, or operating system differs, install the matching PyTorch/DGL builds.

## Repository Structure

```text
NR-Predictor/
|-- NR-2025/
|-- NR_binding/
|   |-- NR_10_endpoints/                 # 10-endpoint binding datasets and splits
|   |-- ml_train.py                      # ML base-model training
|   |-- gnn_train.py                     # Multi-task GNN training
|   |-- gnn_hyper.py                     # GNN hyperparameter optimization
|   |-- ensemble_attention_stacking.py   # Attention-stacking ensemble
|   |-- run_ml_10endpoints.sh
|   |-- run_gnn_10endpoints.sh
|   `-- run_ensemble_attention_10endpoints.sh
|-- NR_function/
|   |-- ago_ant/                         # Functional activity datasets
|   |-- PDB/                              # Receptor structures and Vina configurations
|   |-- models/                           # Model hyperparameters and trained models
|   |-- scripts/
|   |   `-- run_full_pipeline.sh          # Complete functional-activity workflow
|   `-- vina/                             # Bundled Linux x86_64 Vina executable
`-- Picture/
```

## NR-2025

`NR-2025` contains the curated nuclear receptor datasets and data-processing resources used in this project.

Main contents:

- Processed qualitative datasets at 10 uM and 100 uM thresholds.
- Multi-task quantitative dataset.
- External validation dataset primarily collected from JMC literature.
- KNIME processing pipelines used for data cleaning and merging.

## NR_binding

`NR_binding` contains the 10-endpoint ligand-binding training workflow. It trains molecular-feature machine-learning models and multi-task GNN models on both the random and Butina splits, then fits an attention-based stacking ensemble.

### Reproducing Binding Training

Run the following commands from `NR_binding` on a Linux server after activating the environment. The ML and GNN jobs may run concurrently. Start ensemble training only after both base-model jobs have completed.

```bash
cd NR_binding

mkdir -p logs/logs_ml_10endpoints
nohup bash run_ml_10endpoints.sh > logs/logs_ml_10endpoints/run_all.log 2>&1 &

mkdir -p logs/logs_gnn_10endpoints
nohup bash run_gnn_10endpoints.sh > logs/logs_gnn_10endpoints/run_all.log 2>&1 &
```

After the ML and GNN jobs finish, train the random-split attention-stacking ensemble:

```bash
mkdir -p logs/logs_ensemble_10endpoints
nohup python -u ensemble_attention_stacking.py \
    --split random --seed 2878 --device cuda:0 \
    > logs/logs_ensemble_10endpoints/ensemble_random_seed2878.log 2>&1 &
```

`run_ensemble_attention_10endpoints.sh` is also provided to train the random (`seed_2878`, `cuda:0`) and Butina (`seed_3821`, `cuda:1`) ensemble models together.

Training outputs are written under `ml_results_10endpoints/`, `gnn_results_10endpoints/`, and `ensemble_attention_results_10endpoints/`. Each workflow saves task-level and summary evaluation tables in its corresponding `results/` directory.

### GNN Hyperparameter Optimization

`gnn_hyper.py` performs Optuna-based hyperparameter optimization for the GNN base models. Run it before GNN training when new hyperparameters are required. The resulting JSON file should be supplied to `run_gnn_10endpoints.sh` through the expected split-specific filenames:

```bash
cd NR_binding

python -u gnn_hyper.py \
    --csv-path <split_csv> \
    --dataset-root <dataset_root> \
    --params-out best_hyperparameters_<split>_10endpoints.json \
    --device cuda:0
```

For example, use `best_hyperparameters_random_10endpoints.json` for the random split and `best_hyperparameters_butina_10endpoints.json` for the Butina split.

## NR_function

`NR_function` contains the functional activity (agonist/antagonist) model-training workflow for 10 nuclear-receptor targets. The complete pipeline prepares a ligand cache, performs molecular docking and feature assembly, and trains the final XGBoost models for both random and Butina splits.

### Reproducing Functional-Activity Training

Run the full pipeline from `NR_function`:

```bash
cd NR_function
nohup bash scripts/run_full_pipeline.sh >/dev/null 2>&1 &
```

The script writes a timestamped execution log to `NR_function/logs/`, stores trained models under `NR_function/models/`, and saves evaluation results under `NR_function/results/`.

The bundled Vina executable requires a Linux x86_64 environment. Optional resource settings can be supplied when launching the pipeline:

```bash
cd NR_function
PYTHON_BIN=/path/to/env/bin/python LIGAND_WORKERS=12 \
DOCKING_WORKERS=12 CPU_PER_JOB=4 XGB_N_JOBS=12 \
nohup bash scripts/run_full_pipeline.sh >/dev/null 2>&1 &
```

## Notes

- The online server is the entry point for direct prediction: https://lmmd.ecust.edu.cn/nrpre/
- This release provides reproducible training workflows.
- Functional-activity training requires the included receptor PDBQT files, Vina configuration files, and prepared training resources to remain in their expected directories.
