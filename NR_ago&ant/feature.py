import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split

from vina.vina_docking_wrapper import dock

def smiles_to_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(Descriptors.descList) - 1)
    
    try:
        desc_values = [desc(mol) for name, desc in Descriptors.descList if name != 'Ipc']
        descriptors = np.array(desc_values)
        descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
        return descriptors
    except Exception as e:
        print(f"Error calculating descriptors for {smiles}: {e}")
        return np.zeros(len(Descriptors.descList) - 1)

def generate_features_with_docking(
    data_folder='ago_ant',
    mapping_file='vina/mapping.txt',
    pdbqt_folder='vina/pdbqts',
    config_folder='vina/configs',
    output_folder='ago_ant_features',
    vina_exe='vina/vina_1.2.5_linux_x86_64'
):
    os.makedirs(output_folder, exist_ok=True)
    
    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                task, pdb1, pdb2 = parts
                mapping[task] = (pdb1, pdb2)
    
    rdkit_names = [name for name, _ in Descriptors.descList if name != 'Ipc']
    
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        task_name = csv_file.replace('.csv', '')
        print(f"\n{'='*80}")
        print(f"Processing Task: {task_name}")
        print(f"{'='*80}")
        
        if task_name not in mapping:
            print(f"Warning: {task_name} not found in mapping.txt, skipping...")
            continue
        
        pdb1, pdb2 = mapping[task_name]
        print(f"PDB IDs: {pdb1} (agonist), {pdb2} (antagonist)")
        
        df = pd.read_csv(os.path.join(data_folder, csv_file))
        smiles_list = df['SMILES'].tolist()
        labels = df['Label'].values
        
        print("\n[1/4] Generating RDKit descriptors...")
        rdkit_features = np.array([smiles_to_rdkit_descriptors(s) for s in smiles_list])
        print(f"RDKit features shape: {rdkit_features.shape}")

        ligand_folder = os.path.join('vina', 'ligands', task_name)
        ligand_pdbqt_files = [os.path.join(ligand_folder, f"{i}.pdbqt") for i in range(len(smiles_list))]

        valid_ligands = [f for f in ligand_pdbqt_files if os.path.exists(f)]
        if len(valid_ligands) == 0:
            print(f"Error: No ligand PDBQT files found in {ligand_folder}")
            print("Please run prepare_ligands.py first to generate PDBQT files")
            continue

        print(f"Found {len(valid_ligands)}/{len(ligand_pdbqt_files)} ligand PDBQT files")

        print(f"\n[2/4] Docking to {pdb1}...")
        receptor1 = os.path.join(pdbqt_folder, f"{pdb1}.pdbqt")
        config1 = os.path.join(config_folder, f"{pdb1}_vina_config.txt")

        df_dock1 = dock(receptor1, valid_ligands, config1, vina_exe=vina_exe)
        affinity1_dict = dict(zip(df_dock1['ligand_id'], df_dock1['affinity']))

        affinity1 = np.array([affinity1_dict.get(str(i), np.nan) for i in range(len(smiles_list))])

        print(f"\n[3/4] Docking to {pdb2}...")
        receptor2 = os.path.join(pdbqt_folder, f"{pdb2}.pdbqt")
        config2 = os.path.join(config_folder, f"{pdb2}_vina_config.txt")

        df_dock2 = dock(receptor2, valid_ligands, config2, vina_exe=vina_exe)
        affinity2_dict = dict(zip(df_dock2['ligand_id'], df_dock2['affinity']))
        affinity2 = np.array([affinity2_dict.get(str(i), np.nan) for i in range(len(smiles_list))])

        print("\n[4/4] Splitting dataset (8:1:1)...")
        indices = np.arange(len(smiles_list))
        idx_temp, idx_test = train_test_split(
            indices, test_size=0.1, random_state=42, stratify=labels
        )
        idx_train, idx_val = train_test_split(
            idx_temp, test_size=1/9, random_state=42, stratify=labels[idx_temp]
        )
        
        splits = np.empty(len(smiles_list), dtype=object)
        splits[idx_train] = 'train'
        splits[idx_val] = 'val'
        splits[idx_test] = 'test'
        
        feature_df = pd.DataFrame(rdkit_features, columns=rdkit_names)
        feature_df.insert(0, 'SMILES', smiles_list)
        feature_df[f'affinity_{pdb1}'] = affinity1
        feature_df[f'affinity_{pdb2}'] = affinity2
        feature_df['Label'] = labels
        feature_df['split'] = splits
        
        output_file = os.path.join(output_folder, f"{task_name}_features.csv")
        feature_df.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")
        print(f"Shape: {feature_df.shape}")
        print(f"Train: {np.sum(splits=='train')}, Val: {np.sum(splits=='val')}, Test: {np.sum(splits=='test')}")

import time
from datetime import timedelta

if __name__ == '__main__':

    start = time.perf_counter()

    generate_features_with_docking()

    sec = time.perf_counter() - start
    print(f"\n{'='*80}")
    print(f"Elapsed: {timedelta(seconds=int(sec))}")
