import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation


def smiles_to_pdbqt(smiles, output_pdbqt, add_hydrogens=True, verbose=True):
    if verbose:
        print(f"Processing SMILES: {smiles}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    if add_hydrogens:
        mol = Chem.AddHs(mol)

    if verbose:
        print("Generating 3D conformer...")
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    temp_mol = output_pdbqt.replace('.pdbqt', '_temp.mol')
    writer = Chem.SDWriter(temp_mol)
    writer.write(mol)
    writer.close()

    if verbose:
        print("Converting to PDBQT format...")
    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)

    from meeko import PDBQTWriterLegacy
    with open(output_pdbqt, 'w') as f:
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not is_ok:
            raise ValueError(f"Failed to write PDBQT: {error_msg}")
        f.write(pdbqt_string)

    if os.path.exists(temp_mol):
        os.remove(temp_mol)

    if verbose:
        print(f"Ligand PDBQT saved to: {output_pdbqt}")
    return output_pdbqt


def batch_prepare_ligands(smiles_file, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(smiles_file, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(maxsplit=1)
            smiles = parts[0]
            name = parts[1] if len(parts) > 1 else f"ligand_{i}"

            output_pdbqt = os.path.join(output_dir, f"{name}.pdbqt")

            try:
                smiles_to_pdbqt(smiles, output_pdbqt)
                print(f"✓ Successfully prepared: {name}")
            except Exception as e:
                print(f"✗ Failed to prepare {name}: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare ligand from SMILES to PDBQT')
    parser.add_argument('--smiles', type=str, help='SMILES string')
    parser.add_argument('--smiles_file', type=str, help='File containing SMILES (one per line)')
    parser.add_argument('--output', type=str, required=True, help='Output PDBQT file or directory')
    parser.add_argument('--ph', type=float, default=7.4, help='pH value for protonation')

    args = parser.parse_args()

    if args.smiles:
        smiles_to_pdbqt(args.smiles, args.output, args.ph)
    elif args.smiles_file:
        batch_prepare_ligands(args.smiles_file, args.output)
    else:
        print("Error: Must provide either --smiles or --smiles_file")
        parser.print_help()
