import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

from fp_des import extract_features_from_smiles_list


# grid-search spaces
# The current script always uses --params-in and does not run a grid search.
# RF_PARAMS = {
#     'random_state': [42],
#     'max_depth': [i for i in range(1, 10, 2)],
#     'criterion': ['gini'],
#     'class_weight': ['balanced'],
#     'n_estimators': [i for i in range(10, 101, 10)],
# }
#
# SVM_PARAMS = {
#     'random_state': [42],
#     'kernel': ['rbf'],
#     'probability': [True],
#     'class_weight': ['balanced'],
#     'C': [i * 0.1 for i in range(5, 51, 5)],
#     'gamma': ['scale', 'auto', 1e-2, 5e-2, 1e-1, 5e-1],
# }
#
# XGB_PARAMS = {
#     'random_state': [42],
#     'booster': ['gbtree'],
#     'objective': ['binary:logistic'],
#     'max_depth': [i for i in range(1, 10, 2)],
#     'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
#     'n_estimators': [i for i in range(10, 101, 10)],
#     'scale_pos_weight': [None],
# }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train conventional ML models using pre-selected hyperparameters.'
    )
    parser.add_argument('--csv-path', required=True, help='Split CSV containing SMILES, task labels, SOURCES, and split.')
    parser.add_argument('--output-dir', required=True, help='Directory in which models and fixed-threshold results are saved.')
    parser.add_argument('--models', nargs='+', default=['RF', 'SVM', 'XGB'], choices=['RF', 'SVM', 'XGB'])
    parser.add_argument('--features', nargs='+', default=['ecfp', 'pubchem', 'erg', 'rdkit'])
    parser.add_argument(
        '--params-in',
        required=True,
        help='Existing all_best_hyperparameters.json. Hyperparameter search is not performed.',
    )
    return parser.parse_args()


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
    except Exception:
        AUC = np.nan

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        PR_AUC = auc(recall, precision)
    except Exception:
        PR_AUC = np.nan

    return {
        'SE': SE,
        'SP': SP,
        'ACC': ACC,
        'BA': BA,
        'MCC': MCC,
        'AUC': AUC,
        'PR_AUC': PR_AUC,
    }


def load_split_data(df, split, task_name):
    df_split = df[df.iloc[:, -1] == split].copy()
    smiles_list = df_split.iloc[:, 0].tolist()
    labels = df_split[task_name].values

    valid_mask = ~pd.isna(labels)
    smiles_list = [s for i, s in enumerate(smiles_list) if valid_mask[i]]
    labels = labels[valid_mask].astype(float)

    if len(smiles_list) == 0:
        raise ValueError(f'No valid samples for {task_name} in {split} set')
    if len(np.unique(labels)) < 2:
        raise ValueError(f'Only one class for {task_name} in {split} set')

    return smiles_list, labels


def build_feature_matrices(df, task_name, feature_type):
    train_smiles, y_train = load_split_data(df, 'train', task_name)
    val_smiles, y_val = load_split_data(df, 'val', task_name)
    test_smiles, y_test = load_split_data(df, 'test', task_name)

    print(f'  Extracting {feature_type} features...')
    X_train = extract_features_from_smiles_list(train_smiles, feature_type)
    X_val = extract_features_from_smiles_list(val_smiles, feature_type)
    X_test = extract_features_from_smiles_list(test_smiles, feature_type)

    print(f'  Loaded train: {X_train.shape[0]} samples')
    print(f'  Loaded val:   {X_val.shape[0]} samples')
    print(f'  Loaded test:  {X_test.shape[0]} samples')
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_model(model_name, params):
    if model_name == 'RF':
        return RandomForestClassifier(**params)
    if model_name == 'SVM':
        return SVC(**params)
    if model_name == 'XGB':
        return XGBClassifier(**params)
    raise ValueError(f'Unknown model: {model_name}')


def load_best_params(params_path):
    with open(params_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def train_with_fixed_params(best_params, model_name, X_train, y_train, X_val, y_val):
    print('  Training with pre-selected hyperparameters')
    model = create_model(model_name, best_params)
    model.fit(X_train, y_train)

    y_val_proba = model.predict_proba(X_val)[:, 1]
    y_val_pred_05 = (y_val_proba >= 0.5).astype(int)
    val_metrics_05 = calculate_metrics(y_val, y_val_pred_05, y_val_proba)

    print(f"  Validation MCC at the fixed threshold: {val_metrics_05['MCC']:.4f}")
    return val_metrics_05['MCC'], model


def train_and_evaluate(csv_path, output_dir, model_names=None, feature_types=None, params_in=None):
    models_dir = os.path.join(output_dir, 'models')
    results_dir = os.path.join(output_dir, 'results')

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    task_names = df.columns[1:-2].tolist()
    model_names = model_names or ['RF', 'SVM', 'XGB']
    feature_types = feature_types or ['ecfp', 'pubchem', 'erg', 'rdkit']
    fixed_params = load_best_params(params_in)

    print(f'CSV: {csv_path}')
    print(f'Output directory: {output_dir}')
    print(f'Using pre-selected hyperparameters from: {params_in}')
    print(f'Tasks: {task_names}')
    print(f'Total tasks: {len(task_names)}')
    print(f'Models: {model_names}')
    print(f'Features: {feature_types}')

    all_results_05 = []

    print('\n' + '=' * 80)
    print(f'Starting fixed-parameter training: {len(task_names)} Tasks x {len(model_names)} Models x {len(feature_types)} Features')
    print('=' * 80 + '\n')

    for task_name in task_names:
        for model_name in model_names:
            for feature_type in feature_types:
                print(f"\n{'#' * 80}")
                print(f'Task: {task_name} | Model: {model_name} | Feature: {feature_type}')
                print(f"{'#' * 80}")

                try:
                    X_train, y_train, X_val, y_val, X_test, y_test = build_feature_matrices(
                        df, task_name, feature_type
                    )

                    param_key = f'{task_name}_{model_name}_{feature_type}'
                    if param_key not in fixed_params:
                        raise KeyError(f'{param_key} not found in params file: {params_in}')
                    best_params = fixed_params[param_key]
                    best_val_mcc_05, best_model = train_with_fixed_params(
                        best_params, model_name, X_train, y_train, X_val, y_val
                    )

                    print('  Evaluating on test set at the fixed threshold...')
                    y_test_proba = best_model.predict_proba(X_test)[:, 1]
                    y_test_pred_05 = (y_test_proba >= 0.5).astype(int)
                    test_metrics_05 = calculate_metrics(y_test, y_test_pred_05, y_test_proba)

                    base_result = {
                        'Model': model_name,
                        'Task': task_name,
                        'Feature': feature_type,
                    }

                    all_results_05.append({
                        **base_result,
                        **{f'Test_{k}': v for k, v in test_metrics_05.items()},
                    })

                    print('\n  Test Set Results at the fixed threshold:')
                    for metric_name, metric_value in test_metrics_05.items():
                        print(f'    {metric_name}: {metric_value:.4f}')

                    task_model_dir = os.path.join(models_dir, task_name)
                    os.makedirs(task_model_dir, exist_ok=True)
                    model_filename = f'{model_name}_{feature_type}_model.joblib'
                    model_path = os.path.join(task_model_dir, model_filename)
                    joblib.dump(best_model, model_path)
                    print(f'  Model saved: models/{task_name}/{model_filename}')

                except Exception as e:
                    print(f"\n{'!' * 60}")
                    print(f'ERROR: {str(e)}')
                    print(f"{'!' * 60}")
                    import traceback
                    traceback.print_exc()
                    continue

    if not all_results_05:
        print('\nNo successful results to save.')
        return []

    print(f"\n{'=' * 80}")
    print('Saving Summary Results')
    print(f"{'=' * 80}\n")

    save_summary_table(all_results_05, results_dir)
    return all_results_05


def save_summary_table(all_results_05, results_dir):
    numeric_cols = [
        'Test_SE',
        'Test_SP',
        'Test_ACC',
        'Test_BA',
        'Test_MCC',
        'Test_AUC',
        'Test_PR_AUC',
    ]

    summary_05 = pd.DataFrame(all_results_05)
    summary_05[numeric_cols] = summary_05[numeric_cols].round(4)
    summary_05.to_csv(os.path.join(results_dir, 'ml_summary_results.csv'), index=False, float_format='%.4f')
    print('Summary results saved to: results/ml_summary_results.csv')


if __name__ == '__main__':
    start = time.perf_counter()
    args = parse_args()

    train_and_evaluate(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_names=args.models,
        feature_types=args.features,
        params_in=args.params_in,
    )

    end = time.perf_counter()
    elapsed = end - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    print(f'Runtime: {hours} hours {minutes} minutes {seconds:.2f} seconds')
