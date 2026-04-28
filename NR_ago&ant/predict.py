#!/usr/bin/env python3
"""
python predict.py --input smiles.txt --output predictions.csv

python predict.py --input smiles.txt --output predictions.csv --targets AR ESR1

python predict.py --smiles "CCO" "CC(C)O" --output predictions.csv
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import tempfile
import shutil
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy 


from vina.vina_docking_wrapper import dock

TASK_GROUPS = {
    'agonist': ['AR', 'ESR1', 'ESR2', 'GR', 'PPARG', 'RXRA'],
    'antagonist': ['FXR', 'THRB', 'PR'],
    'difference': ['ESRRA']
}

ALL_TASKS = TASK_GROUPS['agonist'] + TASK_GROUPS['antagonist'] + TASK_GROUPS['difference']


def smiles_to_rdkit_descriptors(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(Descriptors.descList) - 1)

    try:
        desc_values = [desc(mol) for name, desc in Descriptors.descList]
        descriptors = np.array(desc_values)
        descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
        return descriptors
    except Exception as e:
        print(f"Warning: Error calculating descriptors for {smiles}: {e}")
        return np.zeros(len(Descriptors.descList) - 1)


def prepare_ligand_pdbqt(smiles, output_path, verbose=True):
    temp_mol = None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            if verbose:
                print(f"    Warning: Invalid SMILES: {smiles}")
            return False

        mol = Chem.AddHs(mol)

        ret = AllChem.EmbedMolecule(mol, randomSeed=42)
        if ret == -1:
            if verbose:
                print(f"    Warning: Failed to generate 3D conformer for {smiles}")
            return False

        AllChem.MMFFOptimizeMolecule(mol)

        temp_mol = output_path.replace('.pdbqt', '_temp.mol')
        writer = Chem.SDWriter(temp_mol)
        writer.write(mol)
        writer.close()

        mol = Chem.MolFromMolFile(temp_mol, removeHs=False)
        if mol is None:
            if verbose:
                print(f"    Warning: Failed to read temporary MOL file")
            if os.path.exists(temp_mol):
                os.remove(temp_mol)
            return False

        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)

        with open(output_path, 'w') as f:
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
            if not is_ok:
                if verbose:
                    print(f"    Warning: Failed to write PDBQT: {error_msg}")
                return False
            f.write(pdbqt_string)

        if temp_mol and os.path.exists(temp_mol):
            os.remove(temp_mol)

        return os.path.exists(output_path)

    except Exception as e:
        if verbose:
            print(f"    Warning: Error converting SMILES to PDBQT: {e}")
        if temp_mol and os.path.exists(temp_mol):
            os.remove(temp_mol)
        return False


def extract_features_for_task(rdkit_features, affinity1, affinity2, task_name):

    if task_name in TASK_GROUPS['agonist']:
        return np.hstack([rdkit_features, affinity1.reshape(-1, 1)])

    elif task_name in TASK_GROUPS['antagonist']:
        return np.hstack([rdkit_features, affinity2.reshape(-1, 1)])

    elif task_name in TASK_GROUPS['difference']:
        diff_score = (affinity1 - affinity2).reshape(-1, 1)
        return np.hstack([rdkit_features, diff_score])

    else:
        raise ValueError(f"Unknown task: {task_name}")


def predict_from_smiles(
    smiles_list,
    targets=None,
    models_dir='ago_ant_results/models',
    models_dir_no_vina='ago_ant_results_no_vina/models',
    mapping_file='vina/mapping.txt',
    pdbqt_folder='vina/pdbqts',
    config_folder='vina/configs',
    vina_exe='vina/vina_1.2.5_linux_x86_64',
    return_proba=True
):

    if targets is None:
        targets = ALL_TASKS
    else:
        invalid_targets = [t for t in targets if t not in ALL_TASKS]
        if invalid_targets:
            raise ValueError(f"Invalid targets: {invalid_targets}. Available: {ALL_TASKS}")

    print(f"{'='*80}")
    print(f"Nuclear Receptor Activity Prediction")
    print(f"{'='*80}")
    print(f"Number of compounds: {len(smiles_list)}")
    print(f"Targets to predict: {', '.join(targets)}")
    print(f"{'='*80}\n")

    mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                task, pdb1, pdb2 = parts
                mapping[task] = (pdb1, pdb2)

    print("[1/3] Generating RDKit descriptors...")
    rdkit_features = np.array([smiles_to_rdkit_descriptors(s) for s in smiles_list])
    print(f"  Shape: {rdkit_features.shape}\n")

    print("[2/3] Preparing ligand PDBQT files...")
    temp_dir = tempfile.mkdtemp(prefix='nr_predict_')
    ligand_pdbqt_files = []
    valid_indices = []
    pdbqt_conversion_status = []  

    for i, smiles in enumerate(smiles_list):
        pdbqt_path = os.path.join(temp_dir, f"ligand_{i}.pdbqt")
        success = prepare_ligand_pdbqt(smiles, pdbqt_path)
        pdbqt_conversion_status.append(success)

        if success:
            ligand_pdbqt_files.append(pdbqt_path)
            valid_indices.append(i) 
        else:
            print(f"  Warning: Failed to convert SMILES {i}: {smiles}")

    print(f"  Successfully prepared {len(ligand_pdbqt_files)}/{len(smiles_list)} ligands")
    print(f"  Will use backup models (descriptors only) for {len(smiles_list) - len(ligand_pdbqt_files)} failed conversions\n")

    docking_cache = {}

    if len(ligand_pdbqt_files) > 0:
        print("[3/4] Performing docking...")

        required_pdbs = set()
        for task in targets:
            if task in mapping:
                pdb1, pdb2 = mapping[task]
                required_pdbs.add(pdb1)
                required_pdbs.add(pdb2)

        print(f"  Docking to {len(required_pdbs)} receptors...")

        for pdb_id in required_pdbs:
            receptor_path = os.path.join(pdbqt_folder, f"{pdb_id}.pdbqt")
            config_path = os.path.join(config_folder, f"{pdb_id}_vina_config.txt")

            if not os.path.exists(receptor_path):
                print(f"    Warning: Receptor file not found: {receptor_path}")
                continue

            if not os.path.exists(config_path):
                print(f"    Warning: Config file not found: {config_path}")
                continue

            print(f"    Docking to {pdb_id}...", end=' ')
            try:
                df_dock = dock(receptor_path, ligand_pdbqt_files, config_path, vina_exe=vina_exe)
                affinity_dict = dict(zip(df_dock['ligand_id'], df_dock['affinity']))

                affinity_full = np.full(len(smiles_list), np.nan)
                for i, valid_idx in enumerate(valid_indices):
                    ligand_id = f"ligand_{valid_idx}"
                    if ligand_id in affinity_dict:
                        affinity_full[valid_idx] = affinity_dict[ligand_id]

                docking_cache[pdb_id] = affinity_full
                print(f"Done")

            except Exception as e:
                print(f"Failed: {e}")
                docking_cache[pdb_id] = np.full(len(smiles_list), np.nan)

        print()
    else:
        print("[3/4] Skipping docking (no valid ligands)")
        print("  Will use backup models (descriptors only) for all molecules\n")

    print("[4/4] Running predictions...")
    predictions = pd.DataFrame({'SMILES': smiles_list})

    for task in targets:
        print(f"  Predicting {task}...", end=' ')

        if task not in mapping:
            print(f"Skipped (not in mapping)")
            predictions[f'{task}_prediction'] = np.nan
            if return_proba:
                predictions[f'{task}_probability'] = np.nan
            predictions[f'{task}_model_type'] = 'N/A'
            continue

        pdb1, pdb2 = mapping[task]
        affinity1 = docking_cache.get(pdb1, np.full(len(smiles_list), np.nan))
        affinity2 = docking_cache.get(pdb2, np.full(len(smiles_list), np.nan))

        task_predictions = []
        task_probabilities = [] if return_proba else None
        task_model_types = []

        use_main_count = 0
        use_backup_count = 0

        for mol_idx in range(len(smiles_list)):
            can_use_main_model = (
                pdbqt_conversion_status[mol_idx] and
                not np.isnan(affinity1[mol_idx]) and
                not np.isnan(affinity2[mol_idx])
            )

            if can_use_main_model:
                model_path = os.path.join(models_dir, f"{task}_model.joblib")
                if not os.path.exists(model_path):
                    task_predictions.append(np.nan)
                    if return_proba:
                        task_probabilities.append(np.nan)
                    task_model_types.append('model_not_found')
                    continue

                model = joblib.load(model_path)
                X = extract_features_for_task(
                    rdkit_features[mol_idx:mol_idx+1],
                    affinity1[mol_idx:mol_idx+1],
                    affinity2[mol_idx:mol_idx+1],
                    task
                )
                print(f"Feature shape: {X.shape}")
                print(f"Model expects: {model.n_features_in_} features")
                model_type = 'descriptors+docking'
                use_main_count += 1

            else:
                model_path = os.path.join(models_dir_no_vina, f"{task}_model.joblib")
                if not os.path.exists(model_path):
                    task_predictions.append(np.nan)
                    if return_proba:
                        task_probabilities.append(np.nan)
                    task_model_types.append('backup_model_not_found')
                    continue

                model = joblib.load(model_path)
                X = rdkit_features[mol_idx:mol_idx+1] 
                model_type = 'descriptors_only'
                use_backup_count += 1

            try:
                pred = model.predict(X)[0] 
                task_predictions.append(pred)

                if return_proba:
                    proba = model.predict_proba(X)[0, 1] 
                    task_probabilities.append(proba)

                task_model_types.append(model_type)

            except Exception as e:
                task_predictions.append(np.nan)
                if return_proba:
                    task_probabilities.append(np.nan)
                task_model_types.append(f'prediction_failed')

        predictions[f'{task}_prediction'] = task_predictions
        if return_proba:
            predictions[f'{task}_probability'] = task_probabilities
        predictions[f'{task}_model_type'] = task_model_types

        active_count = np.sum(np.array(task_predictions) == 1)
        print(f"Done (Active: {active_count}/{len(smiles_list)}, Main: {use_main_count}, Backup: {use_backup_count})")

    shutil.rmtree(temp_dir)

    print(f"\n{'='*80}")
    print("Prediction completed!")
    print(f"{'='*80}\n")

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description='Predict nuclear receptor agonist/antagonist activity from SMILES',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict all targets from file
  python predict.py --input smiles.txt --output predictions.csv

  # Predict specific targets
  python predict.py --input smiles.txt --output predictions.csv --targets AR ESR1

  # Input SMILES directly
  python predict.py --smiles "CCO" "CC(C)O" --output predictions.csv
  python predict.py --smiles "CCO" "Brc1c(Br)c(Br)c(Br)c(Br)c1Br" --output predictions.csv

  # Return binary predictions only (no probabilities)
  python predict.py --input smiles.txt --output predictions.csv --no-proba

Available targets:
  Agonist: AR, ESR1, ESR2, GR, PPARG, RXRA
  Antagonist: FXR, THRB, PR
  Difference: ESRRA
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str,
                             help='Input file containing SMILES (one per line)')
    input_group.add_argument('--smiles', '-s', nargs='+',
                             help='SMILES strings (space separated)')

    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output CSV file path')

    parser.add_argument('--targets', '-t', nargs='+', choices=ALL_TASKS,
                        help='Target nuclear receptors to predict (default: all)')

    parser.add_argument('--models_dir', type=str, default='ago_ant_results/models',
                        help='Directory containing main models (descriptors+docking)')
    parser.add_argument('--models_dir_no_vina', type=str, default='ago_ant_results_no_vina/models',
                        help='Directory containing backup models (descriptors only)')
    parser.add_argument('--mapping_file', type=str, default='vina/mapping.txt',
                        help='Path to mapping.txt')
    parser.add_argument('--pdbqt_folder', type=str, default='vina/pdbqts',
                        help='Path to PDBQT files folder (receptor proteins)')
    parser.add_argument('--config_folder', type=str, default='vina/configs',
                        help='Path to Vina config files folder')
    parser.add_argument('--vina_exe', type=str, default='vina/vina_1.2.5_linux_x86_64',
                        help='Path to Vina executable')

    parser.add_argument('--no-proba', action='store_true',
                        help='Return binary predictions only (no probabilities)')

    args = parser.parse_args()

    if args.input:
        with open(args.input, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    else:
        smiles_list = args.smiles

    if len(smiles_list) == 0:
        print("Error: No SMILES provided")
        return

    predictions = predict_from_smiles(
        smiles_list=smiles_list,
        targets=args.targets,
        models_dir=args.models_dir,
        models_dir_no_vina=args.models_dir_no_vina,
        mapping_file=args.mapping_file,
        pdbqt_folder=args.pdbqt_folder,
        config_folder=args.config_folder,
        vina_exe=args.vina_exe,
        return_proba=not args.no_proba
    )

    if predictions is not None:
        predictions.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}")
        print(f"Shape: {predictions.shape}")

        print("\nPreview (first 5 rows):")
        print(predictions.head())


if __name__ == '__main__':
    main()
