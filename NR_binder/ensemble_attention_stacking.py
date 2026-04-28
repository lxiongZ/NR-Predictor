
import os
import json
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix, precision_recall_curve, auc
)

from dgl_graph import LoadDataset
from gnn_mtl.gnn import create_gnn_model
from fp_des import extract_features_from_smiles_list
from torch_geometric.loader import DataLoader

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

TASK_NAMES = ['BIN_ESRRA', 'BIN_PR', 'BIN_RXRA', 'BIN_ESR1', 'BIN_ESR2',
              'BIN_GR', 'BIN_MR', 'BIN_AR', 'BIN_FXR', 'BIN_PPARG',
              'BIN_THRB', 'BIN_PPARA']

import random

g = None
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    global g
    g = torch.Generator()
    g.manual_seed(seed)
    

def calculate_pos_weights(y_train, num_tasks=12):
    pos_weights = []
    for i in range(num_tasks):
        mask = ~np.isnan(y_train[:, i])
        if mask.sum() == 0:
            pos_weights.append(1.0)
            continue
        
        labels = y_train[mask, i]
        n_neg = np.sum(labels == 0)
        n_pos = np.sum(labels == 1)
        pos_weight = n_neg / (n_pos + 1e-8)
        pos_weights.append(pos_weight)
    
    return torch.FloatTensor(pos_weights)


def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    SE = TP / (TP + FN) if (TP + FN) != 0 else 0
    SP = TN / (TN + FP) if (TN + FP) != 0 else 0
    ACC = accuracy_score(y_true, y_pred)
    BA = balanced_accuracy_score(y_true, y_pred)
    MCC = matthews_corrcoef(y_true, y_pred)

    try:
        AUC = roc_auc_score(y_true, y_prob)
    except:
        AUC = np.nan

    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        PR_AUC = auc(recall, precision)
    except:
        PR_AUC = np.nan

    return {
        'SE': SE, 'SP': SP, 'ACC': ACC,
        'BA': BA, 'MCC': MCC, 'AUC': AUC,
        'PR_AUC': PR_AUC
    }


def load_protein_encodings(csv_path='protein_full_embeddings_12.csv', task_names=None):
    df = pd.read_csv(csv_path, encoding='gbk')

    protein_to_task = {
        'ESRRA': 'BIN_ESRRA', 'PR': 'BIN_PR', 'RXRA': 'BIN_RXRA',
        'ESR1': 'BIN_ESR1', 'ESR2': 'BIN_ESR2', 'GR': 'BIN_GR',
        'MR': 'BIN_MR', 'AR': 'BIN_AR', 'FXR': 'BIN_FXR',
        'PPARG': 'BIN_PPARG', 'THRB': 'BIN_THRB', 'PPARA': 'BIN_PPARA'
    }

    encodings = []
    for task_name in task_names:
        protein_name = [k for k, v in protein_to_task.items() if v == task_name][0]
        row = df[df['protein'] == protein_name]
        if row.empty:
            raise ValueError(f"Protein {protein_name} not found in {csv_path}")
        encoding = row.iloc[0, 2:].values.astype(np.float32)
        encodings.append(encoding)

    return torch.from_numpy(np.array(encodings))


class MLEnsemble:

    def __init__(self, model_type, feature_type, ml_models_dir='ml_results/seed_5673/models'):
        self.model_type = model_type
        self.feature_type = feature_type
        self.ml_models_dir = ml_models_dir
        self.models = {}

        for task_name in TASK_NAMES:
            model_filename = f"{model_type}_{feature_type}_model.joblib"
            model_path = os.path.join(ml_models_dir, task_name, model_filename)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")

            self.models[task_name] = joblib.load(model_path)

    def predict(self, smiles_list):
        X = extract_features_from_smiles_list(smiles_list, self.feature_type)

        probs_list = []
        preds_list = []

        for task_name in TASK_NAMES:
            model = self.models[task_name]
            task_probs = model.predict_proba(X)[:, 1]
            task_preds = (task_probs >= 0.5).astype(int)

            probs_list.append(task_probs)
            preds_list.append(task_preds)

        probs = np.array(probs_list).T
        preds = np.array(preds_list).T

        return probs, preds


class GNNMultiTask:

    def __init__(self, model_name, dataset, gnn_models_dir='gnn_results_protein/seed_5673_esm2/models',
                 params_path='best_hyperparameters.json'):
        self.model_name = model_name
        self.dataset = dataset

        with open(params_path, 'r') as f:
            all_params = json.load(f)
        params = all_params[model_name]

        sample_data = dataset[0]
        in_channels = sample_data.x.size(1)
        edge_dim = sample_data.edge_attr.size(1)

        task_names = TASK_NAMES
        protein_encodings = load_protein_encodings('protein_full_embeddings_12.csv', task_names)

        self.model = create_gnn_model(
            model_name=model_name,
            in_channels=in_channels,
            hidden_channels=params['hidden_channels'],
            out_channels=len(TASK_NAMES),
            edge_dim=edge_dim,
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            num_timesteps=params['num_timesteps'],
            protein_encodings=protein_encodings
        )

        model_path = os.path.join(gnn_models_dir, f'{model_name}_best.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, generator=g)

        all_probs = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)

                if self.model_name == 'dmpnn':
                    out = self.model(data.x, data.edge_index, data.rev_edge_index,
                                   data.edge_attr, data.batch)
                else:
                    out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)

                probs = torch.sigmoid(out)
                all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_preds = (all_probs >= 0.5).astype(int)

        return all_probs, all_preds


class TaskSpecificAttention(nn.Module):

    def __init__(self, num_base_models=6, num_tasks=12, hidden_dim=64, num_heads=4, dropout=0.1):
        super(TaskSpecificAttention, self).__init__()

        self.num_base_models = num_base_models
        self.num_tasks = num_tasks
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        self.task_queries = nn.Parameter(torch.randn(num_tasks, hidden_dim))

        self.key_proj = nn.Linear(1, hidden_dim)
        self.value_proj = nn.Linear(1, hidden_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, return_attention=False):
        batch_size = x.size(0)

        # Reshape: [N, 72] -> [N, 6, 12]
        x = x.view(batch_size, self.num_base_models, self.num_tasks)

        outputs = []
        all_attention_weights = []

        for task_idx in range(self.num_tasks):
            task_preds = x[:, :, task_idx]  # [batch_size, 6]

            keys = self.key_proj(task_preds.unsqueeze(-1))    # [N, 6, hidden_dim]
            values = self.value_proj(task_preds.unsqueeze(-1))  # [N, 6, hidden_dim]

            query = self.task_queries[task_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
            query = query.expand(batch_size, -1, -1)  # [N, 1, hidden_dim]

            attn_output, attn_weights = self.multihead_attn(
                query, keys, values,
                need_weights=True,
                average_attn_weights=True
            )
            # attn_output: [N, 1, hidden_dim]
            # attn_weights: [N, 1, 6]

            task_output = self.output_proj(attn_output.squeeze(1))  # [N, 1]
            outputs.append(task_output)

            if return_attention:
                all_attention_weights.append(attn_weights.squeeze(1))  # [N, 6]

        output = torch.cat(outputs, dim=-1)

        if return_attention:
            attention_weights = torch.stack(all_attention_weights, dim=1)
            return output, attention_weights

        return output


class EnsemblePredictor:

    def __init__(self, dataset, ml_configs, gnn_configs,
                 ml_models_dir='ml_results/seed_5673/models',
                 gnn_models_dir='gnn_results_protein/seed_5673_esm2/models'):
        self.dataset = dataset
        self.ml_models = []
        self.gnn_models = []
        self.attention_meta = None

        self.base_model_names = []

        for model_type, feature_type in ml_configs:
            ml_ensemble = MLEnsemble(model_type, feature_type, ml_models_dir)
            model_name = f"{model_type}_{feature_type}"
            self.ml_models.append((model_name, ml_ensemble))
            self.base_model_names.append(model_name)

        for model_name in gnn_configs:
            gnn_model = GNNMultiTask(model_name, dataset, gnn_models_dir)
            self.gnn_models.append((model_name, gnn_model))
            self.base_model_names.append(model_name.upper())

    def get_base_predictions(self, smiles_list, pyg_dataset):
        all_probs = []

        for name, ml_model in self.ml_models:
            probs, _ = ml_model.predict(smiles_list)
            all_probs.append(probs)

        for name, gnn_model in self.gnn_models:
            probs, _ = gnn_model.predict(pyg_dataset)
            all_probs.append(probs)

        stacked_features = np.concatenate(all_probs, axis=1)

        return stacked_features

    def train_attention_meta(self, train_smiles, train_dataset, train_labels,
                            val_smiles, val_dataset, val_labels,
                            hidden_dim=64, num_heads=4, dropout=0.1,
                            lr=0.001, num_epochs=100, batch_size=32, patience=10):
        print("\n" + "="*70)
        print("Training Attention Meta-Learner (Stacking Layer 2)")
        print("="*70)

        print("Generating base model predictions for training set...")
        train_features = self.get_base_predictions(train_smiles, train_dataset)
        print("Generating base model predictions for validation set...")
        val_features = self.get_base_predictions(val_smiles, val_dataset)

        train_X = torch.FloatTensor(train_features).to(device)
        train_y = torch.FloatTensor(train_labels).to(device)
        val_X = torch.FloatTensor(val_features).to(device)
        val_y = torch.FloatTensor(val_labels).to(device)

        pos_weights = calculate_pos_weights(train_labels, num_tasks=len(TASK_NAMES)).to(device)
        print(f"Pos weights: {pos_weights}")

        self.attention_meta = TaskSpecificAttention(
            num_base_models=len(self.ml_models) + len(self.gnn_models),
            num_tasks=len(TASK_NAMES),
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        ).to(device)

        optimizer = torch.optim.Adam(self.attention_meta.parameters(), lr=lr, weight_decay=1e-5)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            self.attention_meta.train()

            indices = torch.randperm(train_X.size(0), generator=g)
            epoch_loss = 0
            num_batches = 0

            for i in range(0, train_X.size(0), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = train_X[batch_indices]
                batch_y = train_y[batch_indices]

                optimizer.zero_grad()

                logits = self.attention_meta(batch_X)

                loss = 0
                valid_tasks = 0
                for task_idx in range(len(TASK_NAMES)):
                    mask = ~torch.isnan(batch_y[:, task_idx])
                    if mask.sum() > 0:
                        task_loss = F.binary_cross_entropy_with_logits(
                            logits[mask, task_idx],
                            batch_y[mask, task_idx],
                            pos_weight=pos_weights[task_idx]
                        )
                        loss += task_loss
                        valid_tasks += 1

                if valid_tasks > 0:
                    loss = loss / valid_tasks
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1

            self.attention_meta.eval()
            with torch.no_grad():
                val_logits = self.attention_meta(val_X)
                val_loss = 0
                valid_tasks = 0
                for task_idx in range(len(TASK_NAMES)):
                    mask = ~torch.isnan(val_y[:, task_idx])
                    if mask.sum() > 0:
                        task_loss = F.binary_cross_entropy_with_logits(
                            val_logits[mask, task_idx],
                            val_y[mask, task_idx],
                            pos_weight=pos_weights[task_idx]
                        )
                        val_loss += task_loss
                        valid_tasks += 1
                if valid_tasks > 0:
                    val_loss = val_loss / valid_tasks

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | "
                      f"Train Loss: {epoch_loss/num_batches:.4f} | "
                      f"Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.attention_meta.state_dict(), 'ensemble/attention_meta_best_class.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        self.attention_meta.load_state_dict(torch.load('ensemble/attention_meta_best_class.pth'))
        print(f"\nAttention meta-learner training completed. Best Val Loss: {best_val_loss:.4f}")

    def predict_with_attention(self, smiles_list, pyg_dataset, return_attention=True):
        if self.attention_meta is None:
            raise ValueError("Attention meta-learner not trained!")

        print("Generating base model predictions...")
        stacked_features = self.get_base_predictions(smiles_list, pyg_dataset)

        X = torch.FloatTensor(stacked_features).to(device)

        self.attention_meta.eval()
        with torch.no_grad():
            if return_attention:
                logits, attention_weights = self.attention_meta(X, return_attention=True)
                attention_weights = attention_weights.cpu().numpy()  # [N, 12, 6]
            else:
                logits = self.attention_meta(X, return_attention=False)
                attention_weights = None

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        return probs, preds, attention_weights


def evaluate_ensemble(y_true, y_pred, y_prob):
    num_tasks = len(TASK_NAMES)
    task_results = {}

    for i in range(num_tasks):
        task_name = TASK_NAMES[i]
        task_labels = y_true[:, i]
        task_probs = y_prob[:, i]
        task_preds = y_pred[:, i]

        valid_mask = ~np.isnan(task_labels)
        if valid_mask.sum() == 0:
            task_results[task_name] = {
                'SE': np.nan, 'SP': np.nan, 'ACC': np.nan,
                'BA': np.nan, 'MCC': np.nan, 'AUC': np.nan,
                'PR_AUC': np.nan, 'n_samples': 0
            }
            continue

        valid_labels = task_labels[valid_mask]
        valid_probs = task_probs[valid_mask]
        valid_preds = task_preds[valid_mask]

        metrics = calculate_metrics(valid_labels, valid_preds, valid_probs)
        metrics['n_samples'] = int(valid_mask.sum())
        task_results[task_name] = metrics

    # Micro-Average
    all_valid_labels = []
    all_valid_probs = []
    all_valid_preds = []

    for i in range(num_tasks):
        mask = ~np.isnan(y_true[:, i])
        all_valid_labels.extend(y_true[mask, i])
        all_valid_probs.extend(y_prob[mask, i])
        all_valid_preds.extend(y_pred[mask, i])

    all_valid_labels = np.array(all_valid_labels)
    all_valid_probs = np.array(all_valid_probs)
    all_valid_preds = np.array(all_valid_preds)

    micro_metrics = calculate_metrics(all_valid_labels, all_valid_preds, all_valid_probs)
    micro_metrics['n_samples'] = len(all_valid_labels)

    return task_results, micro_metrics


def analyze_attention_importance(attention_weights, base_model_names, task_names, save_dir):
    print("\nAnalyzing attention-based model importance...")

    avg_attention = attention_weights.mean(axis=0)

    importance_df = pd.DataFrame(
        avg_attention,
        index=task_names,
        columns=base_model_names
    )

    importance_df.to_csv(f'{save_dir}/attention_importance_detailed.csv', float_format='%.4f')

    model_avg_importance = importance_df.mean(axis=0).sort_values(ascending=False)
    model_avg_df = pd.DataFrame({
        'Model': model_avg_importance.index,
        'Average_Attention': model_avg_importance.values
    })
    model_avg_df.to_csv(f'{save_dir}/attention_model_importance.csv', index=False, float_format='%.4f')

    task_top_models = []
    for task_idx, task_name in enumerate(task_names):
        task_weights = importance_df.loc[task_name]
        top_model = task_weights.idxmax()
        top_weight = task_weights.max()
        task_top_models.append({
            'Task': task_name,
            'Top_Model': top_model,
            'Attention_Weight': top_weight
        })

    task_top_df = pd.DataFrame(task_top_models)
    task_top_df.to_csv(f'{save_dir}/attention_task_top_models.csv', index=False, float_format='%.4f')

    print(f"Attention importance analysis saved to {save_dir}/")
    print("\nModel Average Importance (across all tasks):")
    print(model_avg_df.to_string(index=False))


def main():
    set_seed(42)

    print("Loading dataset...")
    dataset = LoadDataset(root='./NR_10/seed_5673', raw_filename='NR_with_split_seed5673.csv')

    train_dataset = dataset.get_split_dataset('train')
    val_dataset = dataset.get_split_dataset('val')
    test_dataset = dataset.get_split_dataset('test')

    df = pd.read_csv('./NR_10/seed_5673/raw/NR_with_split_seed5673.csv')

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    train_smiles = train_df['SMILES'].tolist()
    train_labels = train_df[TASK_NAMES].values.astype(float)

    val_smiles = val_df['SMILES'].tolist()
    val_labels = val_df[TASK_NAMES].values.astype(float)

    test_smiles = test_df['SMILES'].tolist()
    test_labels = test_df[TASK_NAMES].values.astype(float)

    print(f"Train: {len(train_smiles)}, Val: {len(val_smiles)}, Test: {len(test_smiles)}")

    ml_configs = [
        ('XGB', 'rdkit'),
        ('SVM', 'pubchem'),
    ]

    gnn_configs = ['dmpnn']

    print("\nLoading base models...")
    ensemble = EnsemblePredictor(
        dataset=dataset,
        ml_configs=ml_configs,
        gnn_configs=gnn_configs
    )

    ensemble.train_attention_meta(
        train_smiles, train_dataset, train_labels,
        val_smiles, val_dataset, val_labels,
        hidden_dim=64,
        num_heads=4,
        dropout=0.1,
        lr=0.001,
        num_epochs=100,
        batch_size=32,
        patience=10
    )

    print("\n" + "="*70)
    print("Testing Attention Stacking on Test Set")
    print("="*70)
    attn_probs, attn_preds, attention_weights = ensemble.predict_with_attention(
        test_smiles, test_dataset, return_attention=True
    )

    attn_task_results, attn_micro_results = evaluate_ensemble(
        test_labels, attn_preds, attn_probs
    )

    print("\nSaving results...")
    save_dir = 'ensemble_results_attention'
    os.makedirs(save_dir, exist_ok=True)

    task_data = []
    for task_name, metrics in attn_task_results.items():
        task_data.append({
            'Task': task_name,
            'SE': metrics['SE'],
            'SP': metrics['SP'],
            'ACC': metrics['ACC'],
            'BA': metrics['BA'],
            'MCC': metrics['MCC'],
            'AUC': metrics['AUC'],
            'PR_AUC': metrics['PR_AUC'],
            'N_samples': metrics['n_samples']
        })

    task_df = pd.DataFrame(task_data)
    numeric_cols = ['SE', 'SP', 'ACC', 'BA', 'MCC', 'AUC', 'PR_AUC']
    task_df[numeric_cols] = task_df[numeric_cols].round(4)
    task_df.to_csv(f'{save_dir}/attention_stacking_task_level.csv',
                   index=False, float_format='%.4f')
    print("Task-level results saved.")

    summary_data = []

    # Macro-Average
    summary_data.append({
        'Metric_Type': 'Macro-Average',
        'SE': task_df['SE'].mean(),
        'SP': task_df['SP'].mean(),
        'ACC': task_df['ACC'].mean(),
        'BA': task_df['BA'].mean(),
        'MCC': task_df['MCC'].mean(),
        'AUC': task_df['AUC'].mean(),
        'PR_AUC': task_df['PR_AUC'].mean()
    })

    # Micro-Average
    summary_data.append({
        'Metric_Type': 'Micro-Average',
        'SE': attn_micro_results['SE'],
        'SP': attn_micro_results['SP'],
        'ACC': attn_micro_results['ACC'],
        'BA': attn_micro_results['BA'],
        'MCC': attn_micro_results['MCC'],
        'AUC': attn_micro_results['AUC'],
        'PR_AUC': attn_micro_results['PR_AUC']
    })

    summary_df = pd.DataFrame(summary_data)
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
    summary_df.to_csv(f'{save_dir}/attention_stacking_summary.csv',
                      index=False, float_format='%.4f')
    print("Summary results saved.")

    analyze_attention_importance(
        attention_weights,
        ensemble.base_model_names,
        TASK_NAMES,
        save_dir
    )

    print("\n" + "="*70)
    print("Attention Stacking ensemble evaluation completed!")
    print("="*70)


if __name__ == '__main__':
    main()
