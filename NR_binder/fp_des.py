import numpy as np
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from pubchemfp import GetPubChemFPs


def smiles_to_ecfps(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    ecfps_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    ecfps_fp = ecfps_gen.GetFingerprintAsNumPy(mol)
    return ecfps_fp


def smiles_to_PubChem(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(881)

    mol2 = Chem.AddHs(mol)
    pubchem_fp = GetPubChemFPs(mol2)
    return pubchem_fp


def smiles_to_Pharmacophore(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(441)

    fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
    return np.array(fp_phaErGfp)


def smiles_to_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(Descriptors.descList))

    try:
        # desc_values = [desc(mol) for _, desc in Descriptors.descList]
        # desc_values = [desc(mol) for name, desc in Descriptors.descList if name != 'Ipc']
        desc_values = [desc(mol) for name, desc in Descriptors.descList if name not in ['Ipc', 'SPS']]
        descriptors = np.array(desc_values)

        descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)

        return descriptors
    except Exception as e:
        print(f"Error calculating descriptors for {smiles}: {e}")
        return np.zeros(len(Descriptors.descList))


def get_feature_by_type(smiles, feature_type='ecfp'):
    feature_type = feature_type.lower()

    if feature_type == 'ecfp':
        return smiles_to_ecfps(smiles)
    elif feature_type == 'pubchem':
        return smiles_to_PubChem(smiles)
    elif feature_type == 'erg':
        return smiles_to_Pharmacophore(smiles)
    elif feature_type == 'rdkit':
        return smiles_to_rdkit_descriptors(smiles)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}. "
                        f"Choose from 'ecfp', 'pubchem', 'erg', 'rdkit'")


def extract_features_from_smiles_list(smiles_list, feature_type='ecfp'):
    features = []
    for smiles in smiles_list:
        feat = get_feature_by_type(smiles, feature_type)
        features.append(feat)

    return np.array(features)