import os
import json
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, confusion_matrix,
    accuracy_score, balanced_accuracy_score, precision_recall_curve, auc
)
from sklearn.model_selection import ParameterGrid, train_test_split
from joblib import Parallel, delayed


xgb_params = {
    'random_state': [42],
    'booster': ['gbtree'],
    'objective': ['binary:logistic'],
    'max_depth': [i for i in range(1, 10, 2)],
    'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
    'n_estimators': [i for i in range(10, 101, 10)],
    'scale_pos_weight': [None]
}

TASK_GROUPS = {
    'agonist': ['AR', 'ESR1', 'ESR2', 'GR', 'PPARG', 'RXRA'],
    'antagonist': ['FXR', 'THRB', 'PR'],
    'difference': ['ESRRA']
}

def calculate_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    SE = TP / (TP + FN) if (TP + FN) != 0 else 0
    SP = TN / (TN + FP) if (TN + FP) != 0 else 0
    ACC = accuracy_score(y_true, y_pred)
    BA = balanced_accuracy_score(y_true, y_pred)
    MCC = matthews_corrcoef(y_true, y_pred)

    try:
        AUC = roc_auc_score(y_true, y_proba)
    except:
        AUC = np.nan

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        PR_AUC = auc(recall, precision)
    except:
        PR_AUC = np.nan

    return {
        'SE': SE, 'SP': SP, 'ACC': ACC,
        'BA': BA, 'MCC': MCC, 'AUC': AUC,
        'PR_AUC': PR_AUC
    }

def extract_features(df, task_name):
    base_features = df.iloc[:, 1:-4].values

    if task_name in TASK_GROUPS['agonist']:
        agonist_score = df.iloc[:, -4].values.reshape(-1, 1)
        X = np.hstack([base_features, agonist_score])
        print(f"  Feature type: Base features + Agonist score")

    elif task_name in TASK_GROUPS['antagonist']:
        antagonist_score = df.iloc[:, -3].values.reshape(-1, 1)
        X = np.hstack([base_features, antagonist_score])
        print(f"  Feature type: Base features + Antagonist score")

    elif task_name in TASK_GROUPS['difference']:
        diff_score = (df.iloc[:, -4] - df.iloc[:, -3]).values.reshape(-1, 1)
        X = np.hstack([base_features, diff_score])
        print(f"  Feature type: Base features + Difference score (Agonist - Antagonist)")

    else:
        raise ValueError(f"Unknown task: {task_name}. Please add it to TASK_GROUPS.")

    y = df.iloc[:, -2].values

    return X, y

def train_and_evaluate_single(params, X_train, y_train, X_val, y_val, scale_pos_weight):
    if params.get('scale_pos_weight') is None:
        params = params.copy()
        params['scale_pos_weight'] = scale_pos_weight

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    scores = calculate_metrics(y_val, y_val_pred, y_val_proba)

    return {
        'params': params,
        'scores': scores,
        'model': model
    }

def grid_search(param_grid, X_train, y_train, X_val, y_val, n_jobs=-1):
    param_list = list(ParameterGrid(param_grid))
    print(f"  Grid search: {len(param_list)} parameter combinations")

    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    results = Parallel(n_jobs=n_jobs)(
        delayed(train_and_evaluate_single)(
            params, X_train, y_train, X_val, y_val, scale_pos_weight
        )
        for params in param_list
    )

    best_result = max(results, key=lambda x: x['scores']['MCC'])
    best_params = best_result['params']
    best_model = best_result['model']
    best_mcc = best_result['scores']['MCC']

    print(f"  Best validation MCC: {best_mcc:.4f}")

    return best_params, best_mcc, best_model

def train_ago_ant_models(
    data_folder='ago_ant_features',
    output_dir='ago_ant_results',
    hyperparams_json=None
):
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, 'models')
    params_dir = os.path.join(output_dir, 'params')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)

    use_preset_params = (hyperparams_json is not None)

    if use_preset_params:
        with open(hyperparams_json, 'r') as f:
            all_hyperparams = json.load(f)
        print(f"\n{'='*80}")
        print(f"MODE: Using preset hyperparameters from: {hyperparams_json}")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"MODE: Grid search enabled")
        print(f"{'='*80}")

    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    results = []
    all_best_params = {}

    for csv_file in csv_files:
        task_name = csv_file.replace('_features.csv', '')
        print(f"\n{'='*80}")
        print(f"Task: {task_name}")
        print(f"{'='*80}")

        df = pd.read_csv(os.path.join(data_folder, csv_file))
        print(f"  Original data size: {len(df)}")

        X, y = extract_features(df, task_name)

        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

        print(f"  Data size after removing NaN: {len(X)}")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.1, random_state=10, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=1/9, random_state=10, stratify=y_temp
        )

        print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

        if use_preset_params:
            print(f"  Using preset hyperparameters for {task_name}...")

            if task_name not in all_hyperparams:
                print(f"  WARNING: No preset hyperparameters found for {task_name}. Skipping...")
                continue

            params = all_hyperparams[task_name].copy()

            if params.get('scale_pos_weight') is None:
                n_neg = np.sum(y_train == 0)
                n_pos = np.sum(y_train == 1)
                params['scale_pos_weight'] = n_neg / n_pos if n_pos > 0 else 1.0

            best_model = XGBClassifier(**params)
            best_model.fit(X_train, y_train)

            y_val_pred = best_model.predict(X_val)
            y_val_proba = best_model.predict_proba(X_val)[:, 1]
            val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
            best_val_mcc = val_metrics['MCC']
            print(f"  Validation MCC: {best_val_mcc:.4f}")

            best_params = params

        else:
            best_params, best_val_mcc, best_model = grid_search(
                xgb_params, X_train, y_train, X_val, y_val, n_jobs=-1
            )

        print(f"  Evaluating on test set...")
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

        print(f"\n  Test Set Results:")
        for metric_name, metric_value in test_metrics.items():
            print(f"    {metric_name}: {metric_value:.4f}")

        model_path = os.path.join(models_dir, f"{task_name}_model.joblib")
        joblib.dump(best_model, model_path)
        print(f"  Model saved: {model_path}")

        if not use_preset_params:
            params_to_save = {}
            for k, v in best_params.items():
                if isinstance(v, (np.integer, np.floating)):
                    params_to_save[k] = v.item()
                else:
                    params_to_save[k] = v
            all_best_params[task_name] = params_to_save

        results.append({
            'Task': task_name,
            'Val_MCC': best_val_mcc,
            'SE': test_metrics['SE'], # Test_
            'SP': test_metrics['SP'],
            'ACC': test_metrics['ACC'],
            'BA': test_metrics['BA'],
            'MCC': test_metrics['MCC'],
            'AUC': test_metrics['AUC'],
            'PR_AUC': test_metrics['PR_AUC']
        })

    df_results = pd.DataFrame(results)
    results_filename = 'summary_results_preset.csv' if use_preset_params else 'summary_results.csv'
    df_results.to_csv(os.path.join(output_dir, results_filename), index=False, float_format='%.4f')
    
    print(f"\n{'='*80}")
    print(f"Summary results saved to: {output_dir}/{results_filename}")

    if not use_preset_params:
        params_path = os.path.join(params_dir, 'best_hyperparameters.json')
        with open(params_path, 'w') as f:
            json.dump(all_best_params, f, indent=4)
        print(f"Best hyperparameters saved to: {params_path}")

    print(f"{'='*80}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train XGBoost models for agonist/antagonist classification')
    parser.add_argument('--data_folder', type=str, default='ago_ant_features',
                        help='Path to data folder')
    parser.add_argument('--output_dir', type=str, default='ago_ant_results/',
                        help='Path to output directory')
    parser.add_argument('--hyperparams', type=str, default=None,
                        help='Path to hyperparameters JSON file (optional)')

    args = parser.parse_args()

    train_ago_ant_models(
        data_folder=args.data_folder,
        output_dir=args.output_dir,
        hyperparams_json=args.hyperparams
    )

#
#    PYTHONWARNINGS=ignore python ml_build_v2.py
#
#    PYTHONWARNINGS=ignore python ml_build_v2.py --hyperparams ago_ant_results/params/best_hyperparameters.json

#    python ml_build_v2.py --data_folder my_data --output_dir my_results
