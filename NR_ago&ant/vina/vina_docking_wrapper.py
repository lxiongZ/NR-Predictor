import os
import subprocess
import tempfile
import pandas as pd
from pathlib import Path
from typing import Union, List


def dock(
    receptor: str,
    ligand: Union[str, List[str]],
    config: str,
    save_best_complex: bool = False,
    output_dir: str = 'docking_results',
    vina_exe: str = 'vina_1.2.5_linux_x86_64'
) -> pd.DataFrame:
    if not os.path.exists(receptor):
        raise FileNotFoundError(f"Receptor not found: {receptor}")
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config file not found: {config}")
    if not os.path.exists(vina_exe):
        raise FileNotFoundError(f"Vina executable not found: {vina_exe}")

    if isinstance(ligand, str):
        ligands = [ligand]
    else:
        ligands = ligand

    for lig in ligands:
        if not os.path.exists(lig):
            raise FileNotFoundError(f"Ligand not found: {lig}")

    if save_best_complex:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    receptor_name = os.path.splitext(os.path.basename(receptor))[0]

    results = []

    with tempfile.TemporaryDirectory() as temp_dir:

        # print(f"\n{'='*60}")
        # print(f"Starting docking: {len(ligands)} ligand(s)")
        # print(f"Using Vina: {vina_exe}")
        # print(f"{'='*60}\n")

        for idx, ligand_path in enumerate(ligands, 1):
            ligand_id = os.path.splitext(os.path.basename(ligand_path))[0]
            
            if idx % 50 == 0 or idx == len(ligands):
                print(f"<{idx}/{len(ligands)}> Processing: {ligand_id}")

            temp_output = os.path.join(temp_dir, f"{ligand_id}_docked.pdbqt")

            try:
                affinity = _run_vina_docking(
                    vina_exe=vina_exe,
                    receptor=receptor,
                    ligand=ligand_path,
                    output=temp_output,
                    config=config
                )

                if affinity is not None:
                    if idx % 50 == 0 or idx == len(ligands):
                        print(f"  ✓ Best affinity: {affinity:.3f} kcal/mol")

                    if save_best_complex:
                        best_pose_file = os.path.join(temp_dir, f"{ligand_id}_best_pose.pdbqt")
                        _extract_best_pose(temp_output, best_pose_file)

                        complex_file = os.path.join(
                            output_dir,
                            f"{ligand_id}_{receptor_name}_complex.pdbqt"
                        )
                        _create_complex(receptor, best_pose_file, complex_file)
                        print(f"  ✓ Complex saved: {complex_file}")
                else:
                    print(f"  ✗ Docking failed")
                    affinity = float('nan')

            except Exception as e:
                print(f"  ✗ Error: {e}")
                affinity = float('nan')

            results.append({
                'ligand_id': ligand_id,
                'affinity': affinity
            })

    df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print(f"Docking completed!")
    print(f"Total: {len(results)} ligands")
    print(f"Success: {df['affinity'].notna().sum()}")
    print(f"Failed: {df['affinity'].isna().sum()}")
    print(f"{'='*60}\n")

    return df


def _run_vina_docking(vina_exe, receptor, ligand, output, config):
    cmd = [
        vina_exe,
        "--config", config,
        "--receptor", receptor,
        "--ligand", ligand,
        "--out", output,
        "--cpu", "20",
        "--seed", "42"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if not os.path.exists(output):
            return None

        affinity = _parse_vina_output(result.stdout)
        return affinity

    except subprocess.CalledProcessError as e:
        print(f"    Vina error: {e.stderr}")
        return None
    except Exception as e:
        print(f"    Unexpected error: {e}")
        return None


def _parse_vina_output(output_text):
    lines = output_text.split('\n')

    for i, line in enumerate(lines):
        if 'mode |   affinity' in line:
            for j in range(i+3, len(lines)):
                parts = lines[j].strip().split()
                if len(parts) >= 4:
                    try:
                        mode_num = int(parts[0])
                        affinity = float(parts[1])
                        if mode_num == 1:
                            return affinity
                    except (ValueError, IndexError):
                        continue
                elif len(parts) == 0:
                    break
            break

    return None


def _extract_best_pose(docked_pdbqt, output_file):
    with open(docked_pdbqt, 'r') as f:
        lines = f.readlines()

    best_pose_lines = []
    in_model_1 = False

    for line in lines:
        if line.startswith('MODEL 1'):
            in_model_1 = True
            best_pose_lines.append(line)
        elif line.startswith('ENDMDL') and in_model_1:
            best_pose_lines.append(line)
            break
        elif in_model_1:
            best_pose_lines.append(line)

    if not best_pose_lines:
        best_pose_lines = lines

    with open(output_file, 'w') as f:
        f.writelines(best_pose_lines)


def _create_complex(receptor_pdbqt, ligand_pdbqt, complex_pdbqt):
    with open(receptor_pdbqt, 'r') as f:
        receptor_lines = f.readlines()

    with open(ligand_pdbqt, 'r') as f:
        ligand_lines = f.readlines()

    ligand_clean = []
    for line in ligand_lines:
        if not line.startswith(('MODEL', 'ENDMDL')):
            ligand_clean.append(line)

    with open(complex_pdbqt, 'w') as f:
        f.writelines(receptor_lines)
        if not receptor_lines[-1].startswith('TER'):
            f.write('TER\n')
        f.writelines(ligand_clean)
        if not ligand_clean[-1].startswith('END'):
            f.write('END\n')


if __name__ == "__main__":
    
    df = dock(
        receptor='receptor.pdbqt',
        ligand='ligand.pdbqt',
        config='1A28_vina_config.txt'
    )
    print(df)
