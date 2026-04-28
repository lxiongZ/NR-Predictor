import warnings
warnings.filterwarnings('ignore')

import json
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, confusion_matrix, precision_recall_curve, auc)
import random
import os
from tqdm import tqdm

from dgl_graph import LoadDataset
from gnn_mtl.gnn import create_gnn_model

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def load_protein_encodings(csv_path='protein_full_embeddings_12.csv', task_names=None):
    df = pd.read_csv(csv_path,encoding='gbk')

    protein_to_task = {
        'ESRRA': 'BIN_ESRRA',
        'PR': 'BIN_PR',
        'RXRA': 'BIN_RXRA',
        'ESR1': 'BIN_ESR1',
        'ESR2': 'BIN_ESR2',
        'GR': 'BIN_GR',
        'MR': 'BIN_MR',
        'AR': 'BIN_AR',
        'FXR': 'BIN_FXR',
        'PPARG': 'BIN_PPARG',
        'THRB': 'BIN_THRB',
        'PPARA': 'BIN_PPARA'
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

def calculate_task_pos_weights(train_dataset, num_tasks=12):
    task_counts = torch.zeros(num_tasks, 2)

    for data in train_dataset:
        labels = data.y.squeeze(0) if data.y.dim() == 2 else data.y
        for i in range(num_tasks):
            if not torch.isnan(labels[i]):
                if labels[i] == 0:
                    task_counts[i, 0] += 1
                else:
                    task_counts[i, 1] += 1

    pos_weights = task_counts[:, 0] / (task_counts[:, 1] + 1e-8)
    return pos_weights

def multitask_weighted_loss(pred, target, pos_weights):
    _ , num_tasks = pred.shape
    total_loss = 0
    valid_tasks = 0

    for i in range(num_tasks):
        mask = ~torch.isnan(target[:, i])

        if mask.sum() == 0:
            continue

        task_loss = F.binary_cross_entropy_with_logits(
            pred[mask, i],
            target[mask, i],
            pos_weight=pos_weights[i].to(pred.device)
        )

        total_loss += task_loss
        valid_tasks += 1

    return total_loss / valid_tasks if valid_tasks > 0 else torch.tensor(0.0)

def train_one_epoch(model, train_loader, optimizer, pos_weights, device, model_name):
    model.train()
    epoch_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        if model_name == 'dmpnn':
            out = model(data.x, data.edge_index, data.rev_edge_index,
                       data.edge_attr, data.batch)
        else:
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        loss = multitask_weighted_loss(out, data.y, pos_weights)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)

def validate_model(model, val_loader, pos_weights, device, model_name):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)

            if model_name == 'dmpnn':
                out = model(data.x, data.edge_index, data.rev_edge_index,
                           data.edge_attr, data.batch)
            else:
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)

            loss = multitask_weighted_loss(out, data.y, pos_weights)
            val_loss += loss.item()

    return val_loss / len(val_loader)

def calculate_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    SE = TP / (TP + FN) if (TP + FN) != 0 else 0  # Sensitivity (Recall)
    SP = TN / (TN + FP) if (TN + FP) != 0 else 0  # Specificity
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

    return SE, SP, ACC, BA, MCC, AUC, PR_AUC

def evaluate_model(model, test_loader, device, model_name, task_names):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)

            if model_name == 'dmpnn':
                out = model(data.x, data.edge_index, data.rev_edge_index,
                           data.edge_attr, data.batch)
            else:
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)

            probs = torch.sigmoid(out)
            all_probs.append(probs.cpu())
            all_labels.append(data.y.cpu())

    all_probs = torch.cat(all_probs, dim=0).numpy()  # [N, num_tasks]
    all_labels = torch.cat(all_labels, dim=0).numpy()

    num_tasks = all_labels.shape[1]
    task_results = {}

    for i in range(num_tasks):
        task_name = task_names[i]
        task_labels = all_labels[:, i]
        task_probs = all_probs[:, i]

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
        valid_preds = (valid_probs >= 0.5).astype(int)

        SE, SP, ACC, BA, MCC, AUC, PR_AUC = calculate_metrics(
            valid_labels, valid_preds, valid_probs
        )

        task_results[task_name] = {
            'SE': SE, 'SP': SP, 'ACC': ACC,
            'BA': BA, 'MCC': MCC, 'AUC': AUC,
            'PR_AUC': PR_AUC, 'n_samples': int(valid_mask.sum())
        }

    all_valid_labels = []
    all_valid_probs = []

    for i in range(num_tasks):
        mask = ~np.isnan(all_labels[:, i])
        all_valid_labels.extend(all_labels[mask, i])
        all_valid_probs.extend(all_probs[mask, i])

    all_valid_labels = np.array(all_valid_labels)
    all_valid_probs = np.array(all_valid_probs)
    all_valid_preds = (all_valid_probs >= 0.5).astype(int)

    micro_SE, micro_SP, micro_ACC, micro_BA, micro_MCC, micro_AUC, micro_PR_AUC = calculate_metrics(
        all_valid_labels, all_valid_preds, all_valid_probs
    )

    micro_results = {
        'SE': micro_SE, 'SP': micro_SP, 'ACC': micro_ACC,
        'BA': micro_BA, 'MCC': micro_MCC, 'AUC': micro_AUC,
        'PR_AUC': micro_PR_AUC, 'n_samples': len(all_valid_labels)
    }

    return task_results, micro_results

def train_and_evaluate(model_name, params, dataset, task_names, in_channels, edge_dim,
                       protein_encodings=None, num_epochs=200, patience=5, save_dir='./gnn_results_protein/seed_5673_esm2'):
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} model")
    print(f"{'='*60}")
    print(f"Parameters: {params}")

    os.makedirs(save_dir, exist_ok=True)

    # train_dataset = [d for d in dataset if d.split == 'train']
    # val_dataset = [d for d in dataset if d.split == 'val']
    # test_dataset = [d for d in dataset if d.split == 'test']

    train_dataset = dataset.get_split_dataset('train')
    val_dataset = dataset.get_split_dataset('val')
    test_dataset = dataset.get_split_dataset('test')

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    pos_weights = calculate_task_pos_weights(train_dataset, num_tasks=len(task_names))
    print(f"Pos weights: {pos_weights.numpy()}")

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    model = create_gnn_model(
        model_name=model_name,
        in_channels=in_channels,
        hidden_channels=params['hidden_channels'],
        out_channels=len(task_names),
        edge_dim=edge_dim,
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        num_timesteps=params['num_timesteps'],
        protein_encodings=protein_encodings
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'],
                                weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params['gamma'])

    best_val_loss = float('inf')
    best_epoch = 0
    early_stopping_counter = 0

    print(f"\nStarting training...")
    for epoch in range(num_epochs):
        if early_stopping_counter <= patience:
            train_loss = train_one_epoch(model, train_loader, optimizer, pos_weights, device, model_name)

            if (epoch + 1) % 5 == 0:
                val_loss = validate_model(model, val_loader, pos_weights, device, model_name)
                print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    early_stopping_counter = 0
                    
                    # torch.save({
                    #     'epoch': epoch,
                    #     'model_state_dict': model.state_dict(),
                    #     'optimizer_state_dict': optimizer.state_dict(),
                    #     'val_loss': val_loss,
                    #     'params': params
                    # }, os.path.join(save_dir, f'{model_name}_best.pth'))
                    
                    save_path = os.path.join(save_dir, 'models')
                    os.makedirs(save_path, exist_ok=True) 
                    torch.save(model.state_dict(), os.path.join(save_path, f'{model_name}_best.pth'))
                    
                    print(f"  → Best model saved (Val Loss: {val_loss:.4f})")
                else:
                    early_stopping_counter += 1

            scheduler.step()
        else:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest model at epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")

    # checkpoint = torch.load(os.path.join(save_dir, f'{model_name}_best.pth'))
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    model.load_state_dict(torch.load(os.path.join(save_path, f'{model_name}_best.pth')))

    print(f"\nEvaluating on test set...")
    task_results, micro_results = evaluate_model(model, test_loader, device, model_name, task_names)

    return task_results, micro_results

def main():
    set_seed(42)

    print("Loading dataset...")
    dataset = LoadDataset(root='./NR_10/seed_5673', raw_filename='NR_with_split_seed5673.csv')

    sample_data = dataset[0]
    in_channels = sample_data.x.size(1)
    edge_dim = sample_data.edge_attr.size(1)
    num_tasks = sample_data.y.size(1) if sample_data.y.dim() == 2 else sample_data.y.size(0)

    task_names = ['BIN_ESRRA', 'BIN_PR', 'BIN_RXRA', 'BIN_ESR1', 'BIN_ESR2',
                  'BIN_GR', 'BIN_MR', 'BIN_AR', 'BIN_FXR', 'BIN_PPARG',
                  'BIN_THRB', 'BIN_PPARA']

    print(f"Dataset: {len(dataset)} molecules")
    print(f"Node features: {in_channels}, Edge features: {edge_dim}")
    print(f"Tasks: {num_tasks}")

    print("\nLoading protein encodings...")
    protein_encodings = load_protein_encodings('protein_full_embeddings_12.csv', task_names)
    print(f"Protein encodings shape: {protein_encodings.shape}")  # [12, 1418]

    with open('best_hyperparameters.json', 'r') as f:
        all_params = json.load(f)

    all_task_results = {}
    all_micro_results = {}
    model_names = ['gt', 'gin', 'gcn', 'gat', 'afp', 'dmpnn']

    for model_name in model_names:
        params = all_params[model_name]
        task_results, micro_results = train_and_evaluate(
            model_name=model_name,
            params=params,
            dataset=dataset,
            task_names=task_names,
            in_channels=in_channels,
            edge_dim=edge_dim,
            protein_encodings=protein_encodings,
            num_epochs=200,
            patience=5
        )
        all_task_results[model_name] = task_results
        all_micro_results[model_name] = micro_results

    print(f"{'='*70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*70}")

    task_data = []
    for model_name, tasks_results in all_task_results.items():
        for task_name, metrics in tasks_results.items():
            task_data.append({
                'Model': model_name.upper(),
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

    save_path = 'gnn_results_protein/seed_5673_esm2/results'
    os.makedirs(save_path, exist_ok=True)

    task_df.to_csv(os.path.join(save_path, 'test_results_task_level.csv'),
               index=False, float_format='%.4f')
    # task_df.to_csv('gnn_results/results/test_results_task_level.csv', index=False, float_format='%.4f')
    print("\nTask-level results saved to: test_results_task_level.csv")

    macro_data = []
    for model_name in model_names:
        model_tasks = task_df[task_df['Model'] == model_name.upper()]
        macro_data.append({
            'Model': model_name.upper(),
            'Metric_Type': 'Macro-Average',
            'SE': model_tasks['SE'].mean(),
            'SP': model_tasks['SP'].mean(),
            'ACC': model_tasks['ACC'].mean(),
            'BA': model_tasks['BA'].mean(),
            'MCC': model_tasks['MCC'].mean(),
            'AUC': model_tasks['AUC'].mean(),
            'PR_AUC': model_tasks['PR_AUC'].mean()
        })

    for model_name, micro_metrics in all_micro_results.items():
        macro_data.append({
            'Model': model_name.upper(),
            'Metric_Type': 'Micro-Average',
            'SE': micro_metrics['SE'],
            'SP': micro_metrics['SP'],
            'ACC': micro_metrics['ACC'],
            'BA': micro_metrics['BA'],
            'MCC': micro_metrics['MCC'],
            'AUC': micro_metrics['AUC'],
            'PR_AUC': micro_metrics['PR_AUC']
        })

    summary_df = pd.DataFrame(macro_data)
    summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
    summary_df.to_csv(os.path.join(save_path, 'test_results_summary.csv'),
               index=False, float_format='%.4f')
    # summary_df.to_csv('gnn_results/results/test_results_summary.csv', index=False, float_format='%.4f')
    print("Summary results (Macro & Micro) saved to: test_results_summary.csv")

    # print(f"\n{'='*70}")
    # print("Macro-Average Results (Task-wise average):")
    # print(f"{'='*70}")
    # macro_df = summary_df[summary_df['Metric_Type'] == 'Macro-Average']
    # print(macro_df[['Model', 'SE', 'SP', 'ACC', 'BA', 'MCC', 'AUC', 'PR_AUC']].to_string(index=False))

    # print(f"\n{'='*70}")
    # print("Micro-Average Results (Sample-wise average):")
    # print(f"{'='*70}")
    # micro_df = summary_df[summary_df['Metric_Type'] == 'Micro-Average']
    # print(micro_df[['Model', 'SE', 'SP', 'ACC', 'BA', 'MCC', 'AUC', 'PR_AUC']].to_string(index=False))

    # print(f"\n{'='*70}")
    # print("Training and evaluation completed!")
    # print(f"{'='*70}")

import time
if __name__ == '__main__':
    start = time.perf_counter()
    main()

    end = time.perf_counter()
    elapsed = end - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    print(f"Runtime: {hours} hours {minutes} minutes {seconds:.2f} seconds")



