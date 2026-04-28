import warnings
warnings.filterwarnings('ignore')

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier # pip install xgboost==3.0.5
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_recall_curve, auc
)
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

from fp_des import extract_features_from_smiles_list

warnings.filterwarnings('ignore')


RF_params = {
    'random_state': [42],
    'max_depth': [i for i in range(1, 10, 2)],
    'criterion': ['gini'],
    'class_weight': ['balanced'],
    'n_estimators': [i for i in range(10, 101, 10)]
}

SVM_params = {
    'random_state': [42],
    'kernel': ['rbf'],
    'probability': [True],
    'class_weight': ['balanced'],
    'C': [i * 0.1 for i in range(5, 51, 5)],
    'gamma': ['scale', 'auto', 1e-2, 5e-2, 1e-1, 5e-1]
}

xgb_params = {
    'random_state': [42],
    'booster': ['gbtree'],
    'objective': ['binary:logistic'],
    'max_depth': [i for i in range(1, 10, 2)],
    'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
    'n_estimators': [i for i in range(10, 101, 10)],
    'scale_pos_weight': [None]
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



def load_data(csv_path, split, task_name, feature_type='ecfp'):
    df = pd.read_csv(csv_path)

    df_split = df[df.iloc[:, -1] == split].copy()

    smiles_list = df_split.iloc[:, 0].tolist()
    labels = df_split[task_name].values

    valid_mask = ~pd.isna(labels)
    smiles_list = [s for i, s in enumerate(smiles_list) if valid_mask[i]]
    labels = labels[valid_mask].astype(float)

    if len(smiles_list) == 0:
        raise ValueError(f"No valid samples for {task_name} in {split} set")

    X = extract_features_from_smiles_list(smiles_list, feature_type)
    y = labels

    print(f"  Loaded {split}: {X.shape[0]} samples")
    return X, y



def train_and_evaluate_single(params, model_name, X_train, y_train, X_val, y_val, scale_pos_weight=None):
    if model_name == 'XGB' and params.get('scale_pos_weight') is None:
        params = params.copy()
        params['scale_pos_weight'] = scale_pos_weight

    if model_name == 'RF':
        model = RandomForestClassifier(**params)
    elif model_name == 'SVM':
        model = SVC(**params)
    elif model_name == 'XGB':
        model = XGBClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    scores = calculate_metrics(y_val, y_val_pred, y_val_proba)

    return {
        'params': params,
        'scores': scores,
        'model': model
    }


def grid_search(model_name, param_grid, X_train, y_train, X_val, y_val, n_jobs=30):
    param_list = list(ParameterGrid(param_grid))
    print(f"  Grid search: {len(param_list)} parameter combinations")

    scale_pos_weight = None
    if model_name == 'XGB':
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    results = Parallel(n_jobs=n_jobs)(
        delayed(train_and_evaluate_single)(
            params, model_name, X_train, y_train, X_val, y_val, scale_pos_weight
        )
        for params in param_list
    )

    best_result = max(results, key=lambda x: x['scores']['MCC'])
    best_params = best_result['params']
    best_model = best_result['model']
    best_mcc = best_result['scores']['MCC']

    print(f"  Best validation MCC: {best_mcc:.4f}")

    return best_params, best_mcc, best_model



def train_and_evaluate(csv_path, output_dir='ml_results'):
    models_dir = os.path.join(output_dir, 'models')
    params_dir = os.path.join(output_dir, 'params')
    results_dir = os.path.join(output_dir, 'results')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    task_names = df.columns[1:-2].tolist()

    print(f"Tasks: {task_names}")
    print(f"Total tasks: {len(task_names)}")

    models = {
        'RF': RF_params,
        'SVM': SVM_params,
        'XGB': xgb_params
    }

    feature_types = ['ecfp', 'pubchem', 'erg', 'rdkit']

    all_results = []
    all_best_params = {}

    print("\n" + "="*80)
    print(f"Starting Training Pipeline: {len(task_names)} Tasks × 3 Models × 4 Features")
    print("="*80 + "\n")

    for task_name in task_names:
        for model_name, param_grid in models.items():
            for feature_type in feature_types:
                print(f"\n{'#'*80}")
                print(f"Task: {task_name} | Model: {model_name} | Feature: {feature_type}")
                print(f"{'#'*80}")

                try:
                    X_train, y_train = load_data(csv_path, 'train', task_name, feature_type)
                    X_val, y_val = load_data(csv_path, 'val', task_name, feature_type)
                    X_test, y_test = load_data(csv_path, 'test', task_name, feature_type)

                    best_params, best_val_mcc, best_model = grid_search(
                        model_name, param_grid, X_train, y_train, X_val, y_val, n_jobs=30
                    )

                    print(f"  Evaluating on test set...")
                    y_test_pred = best_model.predict(X_test)
                    y_test_proba = best_model.predict_proba(X_test)[:, 1]
                    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

                    result = {
                        'task': task_name,
                        'model': model_name,
                        'feature_type': feature_type,
                        'best_params': best_params,
                        'val_mcc': best_val_mcc,
                        'test_metrics': test_metrics
                    }
                    all_results.append(result)

                    print(f"\n  Test Set Results:")
                    for metric_name, metric_value in test_metrics.items():
                        print(f"    {metric_name}: {metric_value:.4f}")

                    task_model_dir = os.path.join(models_dir, task_name)
                    os.makedirs(task_model_dir, exist_ok=True)
                    model_filename = f"{model_name}_{feature_type}_model.joblib"
                    model_path = os.path.join(task_model_dir, model_filename)
                    joblib.dump(best_model, model_path)
                    print(f"  Model saved: models/{task_name}/{model_filename}")

                    param_key = f"{task_name}_{model_name}_{feature_type}"
                    params_to_save = {}
                    for k, v in best_params.items():
                        if isinstance(v, (np.integer, np.floating)):
                            params_to_save[k] = v.item()
                        else:
                            params_to_save[k] = v
                    all_best_params[param_key] = params_to_save

                except Exception as e:
                    print(f"\n{'!'*60}")
                    print(f"ERROR: {str(e)}")
                    print(f"{'!'*60}")
                    import traceback
                    traceback.print_exc()
                    continue

    if not all_results:
        print("\n⚠️ No successful results to save!")
        return []

    print(f"\n{'='*80}")
    print("Saving Summary Results")
    print(f"{'='*80}\n")

    summary_df = pd.DataFrame([
        {
            'Model': r['model'],
            'Task': r['task'],
            'Feature': r['feature_type'],
            'Val_MCC': r['val_mcc'],
            'Test_SE': r['test_metrics']['SE'],
            'Test_SP': r['test_metrics']['SP'],
            'Test_ACC': r['test_metrics']['ACC'],
            'Test_BA': r['test_metrics']['BA'],
            'Test_MCC': r['test_metrics']['MCC'],
            'Test_AUC': r['test_metrics']['AUC'],
            'Test_PR_AUC': r['test_metrics']['PR_AUC']
        }
        for r in all_results
    ])

    numeric_cols = ['Val_MCC', 'Test_SE', 'Test_SP', 'Test_ACC', 'Test_BA', 'Test_MCC', 'Test_AUC', 'Test_PR_AUC']
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)

    summary_path = os.path.join(results_dir, 'ml_summary_results.csv')
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"Summary results saved to: results/ml_summary_results.csv")

    if all_best_params:
        all_params_path = os.path.join(params_dir, 'all_best_hyperparameters.json')
        with open(all_params_path, 'w') as f:
            json.dump(all_best_params, f, indent=4)
        print(f"All best hyperparameters saved to: all_best_hyperparameters.json")
        print(f"  Total configurations: {len(all_best_params)}")

    return all_results


import time

if __name__ == '__main__':
    start = time.perf_counter()

    csv_path = 'NR_10/seed_5673/raw/NR_with_split_seed5673.csv'
    output_dir = 'ml_results/seed_5673'

    results = train_and_evaluate(csv_path, output_dir)

    end = time.perf_counter()
    elapsed = end - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    print(f"Runtime: {hours} hours {minutes} minutes {seconds:.2f} seconds")

# PYTHONWARNINGS=ignore nohup python -u ml_train.py > logs/ml_hpyer.log 2>&1 &

# [1] 93182 15152
# pkill -u "$USER" -f python


