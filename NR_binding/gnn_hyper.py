import warnings
warnings.filterwarnings('ignore')

import argparse
import optuna
from optuna.samplers import TPESampler
import json
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import random
import os
import shutil
from pathlib import Path

from dgl_graph import LoadDataset
from gnn_mtl.gnn import create_gnn_model

TASK_NAMES = ['BIN_ESRRA', 'BIN_PR', 'BIN_RXRA', 'BIN_ESR1', 'BIN_ESR2',
              'BIN_GR', 'BIN_AR', 'BIN_FXR', 'BIN_PPARG', 'BIN_THRB']


def parse_args():
    parser = argparse.ArgumentParser(description='GNN hyperparameter search for NR binding prediction.')
    parser.add_argument('--csv-path', required=True, help='Split CSV containing SMILES, task labels, SOURCES, and split.')
    parser.add_argument('--dataset-root', required=True, help='PyG dataset root. The CSV will be copied to <root>/raw/.')
    parser.add_argument('--raw-filename', default=None, help='Raw filename under <dataset-root>/raw/. Defaults to basename(csv-path).')
    parser.add_argument('--params-out', default='best_hyperparameters_10endpoints.json', help='Output JSON for best GNN hyperparameters.')
    parser.add_argument('--protein-csv', default='protein_full_embeddings_12.csv', help='Protein embedding CSV path.')
    parser.add_argument('--n-trials', type=int, default=30, help='Optuna trials per GNN model.')
    parser.add_argument('--model-names', nargs='+', default=['gt', 'gin', 'gcn', 'gat', 'afp', 'dmpnn'])
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, cuda:1, ...')
    parser.add_argument('--force-reprocess', action='store_true', help='Delete <dataset-root>/processed before loading.')
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_arg)


def prepare_dataset_root(csv_path, dataset_root, raw_filename=None, force_reprocess=False):
    csv_path = Path(csv_path).resolve()
    dataset_root = Path(dataset_root)
    raw_filename = raw_filename or csv_path.name
    raw_dir = dataset_root / 'raw'
    processed_dir = dataset_root / 'processed'

    raw_dir.mkdir(parents=True, exist_ok=True)
    if force_reprocess and processed_dir.exists():
        shutil.rmtree(processed_dir)

    dest = raw_dir / raw_filename
    if csv_path != dest.resolve():
        shutil.copy2(csv_path, dest)

    return str(dataset_root), raw_filename

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
    df = pd.read_csv(csv_path, encoding='gbk')

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

def calculate_task_pos_weights(train_dataset, num_tasks=None):
    if num_tasks is None:
        sample_labels = train_dataset[0].y.squeeze(0) if train_dataset[0].y.dim() == 2 else train_dataset[0].y
        num_tasks = sample_labels.numel()
    task_counts = torch.zeros(num_tasks, 2)  # [num_tasks, 2] -> [neg, pos]

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

    return total_loss / valid_tasks if valid_tasks > 0 else pred.new_tensor(0.0)

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

def objective(trial, train_dataset, val_dataset, model_name, in_channels, edge_dim,
              protein_encodings, task_names, device):
    
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 5)
    dropout = trial.suggest_categorical('dropout', [0.2, 0.5])
    lr = trial.suggest_categorical('lr', [0.001, 0.0005, 0.0001])
    weight_decay = trial.suggest_categorical('weight_decay', [1e-5, 1e-4, 1e-3])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    gamma = trial.suggest_categorical('gamma', [0.95, 0.99, 1])
    num_timesteps = trial.suggest_int('num_timesteps', 1, 3)
    trial_protein_encodings = protein_encodings.detach().clone()

    model = create_gnn_model(
        model_name=model_name,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=len(task_names),
        edge_dim=edge_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_timesteps=num_timesteps,
        protein_encodings=trial_protein_encodings
    )
    model.to(device)

    pos_weights = calculate_task_pos_weights(train_dataset, num_tasks=len(task_names))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    best_loss = float('inf')
    early_stopping_counter = 0
    patience = 5

    for epoch in range(200):
        if early_stopping_counter <= patience:
            _ = train_one_epoch(model, train_loader, optimizer, pos_weights, device, model_name)

            if (epoch + 1) % 5 == 0:
                val_loss = validate_model(model, val_loader, pos_weights, device, model_name)
                print(f"Trial {trial.number} | Epoch [{epoch + 1}/200] | Val Loss: {val_loss:.4f}") # epoch

                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

            scheduler.step()
        else:
            print(f"Trial {trial.number} | Early stopping")
            break

    return best_loss

from tqdm import tqdm
def hyperparameter_search(dataset, model_names, in_channels, edge_dim, protein_encodings,
                          task_names, device, n_trials=30,
                          params_out='best_hyperparameters_10endpoints.json'):

    train_dataset = dataset.get_split_dataset('train')
    val_dataset = dataset.get_split_dataset('val')

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    best_params = {}

    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Searching hyperparameters for {model_name.upper()}")
        print(f"{'='*50}")

        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction='minimize',
            study_name=f"NR_multitask_{model_name}_protein",
            sampler=sampler
        )

        study.optimize(
            lambda trial: objective(trial, train_dataset, val_dataset,
                                   model_name, in_channels, edge_dim, protein_encodings,
                                   task_names, device),
            n_trials=n_trials
        )

        best_params[model_name] = study.best_params
        print(f"\nBest params for {model_name}: {study.best_params}")
        print(f"Best validation loss: {study.best_value:.4f}")

    # 保存结果
    with open(params_out, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"\n{'='*50}")
    print("Hyperparameter search completed!")
    print(f"Results saved to: {params_out}")
    print(f"{'='*50}")

import time
if __name__ == '__main__':
    
    start = time.perf_counter()
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    set_seed(42)

    dataset_root, raw_filename = prepare_dataset_root(
        args.csv_path,
        args.dataset_root,
        raw_filename=args.raw_filename,
        force_reprocess=args.force_reprocess
    )
    dataset = LoadDataset(root=dataset_root, raw_filename=raw_filename)

    sample_data = dataset[0]
    in_channels = sample_data.x.size(1)
    edge_dim = sample_data.edge_attr.size(1)
    num_tasks = sample_data.y.size(1) if sample_data.y.dim() == 2 else sample_data.y.size(0)

    print(f"Dataset loaded: {len(dataset)} molecules")
    print(f"Node feature dim: {in_channels}, Edge feature dim: {edge_dim}")
    print(f"Number of tasks: {num_tasks}")
    if num_tasks != len(TASK_NAMES):
        raise ValueError(f"Dataset has {num_tasks} tasks, but TASK_NAMES has {len(TASK_NAMES)} tasks: {TASK_NAMES}")

    print("\nLoading protein encodings...")
    protein_encodings = load_protein_encodings(args.protein_csv, TASK_NAMES)
    print(f"Protein encodings shape: {protein_encodings.shape}")

    hyperparameter_search(
        dataset=dataset,
        model_names=args.model_names,
        in_channels=in_channels,
        edge_dim=edge_dim,
        protein_encodings=protein_encodings,
        task_names=TASK_NAMES,
        device=device,
        n_trials=args.n_trials,
        params_out=args.params_out
    )
    
    end = time.perf_counter()
    elapsed = end - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = elapsed % 60

    print(f"Runtime: {hours} hours {minutes} minutes {seconds:.2f} seconds")
