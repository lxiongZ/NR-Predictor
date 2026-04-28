import torch
import os
import sys
import pandas as pd
from rdkit import Chem
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CURRENT_DIR = str(BASE_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

os.chdir(BASE_DIR)

DATASET_ROOT = BASE_DIR / 'NR_10' / 'seed_5673'
ML_MODELS_DIR = BASE_DIR / 'ml_results' / 'seed_5673' / 'models'
GNN_MODELS_DIR = BASE_DIR / 'gnn_results_protein' / 'seed_5673_esm2' / 'models'
ATTENTION_MODEL_PATH = BASE_DIR / 'ensemble' / 'attention_meta_best_class.pth'

import ensemble_attention_stacking as ensemble_module

ensemble_module.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EnsemblePredictor = ensemble_module.EnsemblePredictor
TaskSpecificAttention = ensemble_module.TaskSpecificAttention
TASK_NAMES = ensemble_module.TASK_NAMES
set_seed = ensemble_module.set_seed
device = ensemble_module.device

from dgl_graph import LoadDataset, mol_to_graph_data_obj_simple

import warnings
warnings.filterwarnings('ignore')

set_seed(42) 

print("Initialize the dataset and model...")
dataset = LoadDataset(root=str(DATASET_ROOT), raw_filename='NR_with_split_seed5673.csv')

ml_configs = [('XGB', 'rdkit'), ('SVM', 'pubchem')] 
gnn_configs = ['dmpnn'] 

ensemble = EnsemblePredictor(
    dataset=dataset,
    ml_configs=ml_configs,
    gnn_configs=gnn_configs,
    ml_models_dir=str(ML_MODELS_DIR),
    gnn_models_dir=str(GNN_MODELS_DIR)
)

ensemble.attention_meta = TaskSpecificAttention(
    num_base_models=3, num_tasks=12, hidden_dim=64, num_heads=4, dropout=0.1
).to(device)

ensemble.attention_meta.load_state_dict(torch.load(str(ATTENTION_MODEL_PATH), map_location=device))
ensemble.attention_meta.eval()

def canonicalize_smiles(smiles_list):
    canonical = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Warning: invalid SMILES - {smi}")
            canonical.append(None)
        else:
            canonical.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    return canonical

def create_pyg_dataset(smiles_list):
    data_objects = []
    valid_smiles = []
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            data = mol_to_graph_data_obj_simple(mol)
            data_objects.append(data)
            valid_smiles.append(smi)
    
    return data_objects, valid_smiles

def predict_smiles(smiles_list):

    canonical_smiles = canonicalize_smiles(smiles_list)
    
    valid_idx = [i for i, smi in enumerate(canonical_smiles) if smi is not None]
    valid_smiles = [canonical_smiles[i] for i in valid_idx]
    
    if not valid_smiles:
        print("error")
        return None
    
    print(f"Processing {len(valid_smiles)} valid SMILES...")
    
    pyg_dataset, _ = create_pyg_dataset(valid_smiles)
    
    probs , preds, attention_weights = ensemble.predict_with_attention(
        valid_smiles, pyg_dataset, return_attention=False
    )
    
    results = pd.DataFrame({'SMILES': valid_smiles})
    for i, task in enumerate(TASK_NAMES):
        results[f'{task}_pred'] = preds[:, i]
        results[f'{task}_proba'] = probs[:, i]

    return results

if __name__ == '__main__':
    test_smiles = [
        "OC1=CC=C2C(SC(C3=CC=C(O)C=C3)=C2C(C4=CC=C(OCCN5CCCCC5)C=C4)=O)=C1",
        "OC1=C(C)C=C2C(SC(C3=CC=C(O)C=C3)=C2C(C4=CC=C(OCCN5CCCCC5)C=C4)=O)=C1C"
    ]
    results = predict_smiles(test_smiles)
    print(results)
