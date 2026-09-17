"""Small AutoDock Vina wrapper used by the feature pipeline."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd


def dock_one(
    receptor: str,
    ligand: str,
    config: str,
    vina_exe: str = "vina/vina_1.2.5_linux_x86_64",
    cpu: int = 1,
    seed: int = 42,
    output_file: Optional[str] = None,
    keep_output: bool = False,
) -> Dict[str, object]:
    """Run one Vina docking job and return a serializable result dict."""
    ligand_id = Path(ligand).stem
    result: Dict[str, object] = {
        "ligand_id": ligand_id,
        "affinity": float("nan"),
        "status": "failed",
        "error": "",
    }

    for path, label in (
        (receptor, "receptor"),
        (ligand, "ligand"),
        (config, "config"),
        (vina_exe, "vina executable"),
    ):
        if not os.path.exists(path):
            result["error"] = f"Missing {label}: {path}"
            return result

    try:
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            affinity, run_error = _run_vina_docking_with_error(
                vina_exe=vina_exe,
                receptor=receptor,
                ligand=ligand,
                output=output_file,
                config=config,
                cpu=cpu,
                seed=seed,
            )
            if not keep_output and os.path.exists(output_file):
                os.remove(output_file)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_output = os.path.join(temp_dir, f"{ligand_id}_docked.pdbqt")
                affinity, run_error = _run_vina_docking_with_error(
                    vina_exe=vina_exe,
                    receptor=receptor,
                    ligand=ligand,
                    output=temp_output,
                    config=config,
                    cpu=cpu,
                    seed=seed,
                )

        if affinity is None:
            result["error"] = run_error or "Vina did not return a valid affinity"
            return result

        result["affinity"] = affinity
        result["status"] = "success"
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


def dock(
    receptor: str,
    ligand: Union[str, List[str]],
    config: str,
    save_best_complex: bool = False,
    output_dir: str = "docking_results",
    vina_exe: str = "vina/vina_1.2.5_linux_x86_64",
    cpu: int = 1,
    seed: int = 42,
    progress_every: int = 10,
) -> pd.DataFrame:
    """
    Sequential compatibility wrapper.

    Returns a DataFrame with ligand_id, affinity, status, and error. Use
    feature.py for parallel docking and resumable score caches.
    """
    if isinstance(ligand, str):
        ligands = [ligand]
    else:
        ligands = list(ligand)

    if save_best_complex:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    receptor_name = Path(receptor).stem
    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, ligand_path in enumerate(ligands, 1):
            ligand_id = Path(ligand_path).stem
            output_file = os.path.join(temp_dir, f"{ligand_id}_docked.pdbqt")
            result = dock_one(
                receptor=receptor,
                ligand=ligand_path,
                config=config,
                vina_exe=vina_exe,
                cpu=cpu,
                seed=seed,
                output_file=output_file,
                keep_output=save_best_complex,
            )

            if save_best_complex and result["status"] == "success":
                best_pose_file = os.path.join(temp_dir, f"{ligand_id}_best_pose.pdbqt")
                _extract_best_pose(output_file, best_pose_file)
                complex_file = os.path.join(
                    output_dir,
                    f"{ligand_id}_{receptor_name}_complex.pdbqt",
                )
                _create_complex(receptor, best_pose_file, complex_file)

            results.append(result)
            if progress_every and (idx % progress_every == 0 or idx == len(ligands)):
                print(f"[{idx}/{len(ligands)}] docked against {receptor_name}")

    df = pd.DataFrame(results)
    print(
        f"Docking completed for {receptor_name}: "
        f"{df['status'].eq('success').sum()} success, "
        f"{df['status'].ne('success').sum()} failed"
    )
    return df


def _run_vina_docking(
    vina_exe: str,
    receptor: str,
    ligand: str,
    output: str,
    config: str,
    cpu: int = 1,
    seed: int = 42,
) -> Optional[float]:
    affinity, _ = _run_vina_docking_with_error(
        vina_exe=vina_exe,
        receptor=receptor,
        ligand=ligand,
        output=output,
        config=config,
        cpu=cpu,
        seed=seed,
    )
    return affinity


def _run_vina_docking_with_error(
    vina_exe: str,
    receptor: str,
    ligand: str,
    output: str,
    config: str,
    cpu: int = 1,
    seed: int = 42,
) -> tuple[Optional[float], str]:
    cmd = [
        vina_exe,
        "--config",
        config,
        "--receptor",
        receptor,
        "--ligand",
        ligand,
        "--out",
        output,
        "--cpu",
        str(cpu),
        "--seed",
        str(seed),
    ]

    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        return None, message[:2000]

    if not os.path.exists(output):
        return None, "Vina output file was not created"

    affinity = _parse_vina_output(completed.stdout)
    if affinity is None:
        return None, "Could not parse affinity from Vina stdout"
    return affinity, ""


def _parse_vina_output(output_text: str) -> Optional[float]:
    """Parse the best affinity from Vina stdout."""
    lines = output_text.splitlines()
    for i, line in enumerate(lines):
        if "mode |   affinity" not in line:
            continue
        for row in lines[i + 3 :]:
            parts = row.strip().split()
            if not parts:
                break
            if len(parts) < 2:
                continue
            try:
                mode_num = int(parts[0])
                affinity = float(parts[1])
            except ValueError:
                continue
            if mode_num == 1:
                return affinity
        break
    return None


def _extract_best_pose(docked_pdbqt: str, output_file: str) -> None:
    with open(docked_pdbqt, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    best_pose_lines = []
    in_model_1 = False
    for line in lines:
        if line.startswith("MODEL 1"):
            in_model_1 = True
            best_pose_lines.append(line)
        elif line.startswith("ENDMDL") and in_model_1:
            best_pose_lines.append(line)
            break
        elif in_model_1:
            best_pose_lines.append(line)

    if not best_pose_lines:
        best_pose_lines = lines

    with open(output_file, "w", encoding="utf-8") as handle:
        handle.writelines(best_pose_lines)


def _create_complex(receptor_pdbqt: str, ligand_pdbqt: str, complex_pdbqt: str) -> None:
    with open(receptor_pdbqt, "r", encoding="utf-8") as handle:
        receptor_lines = handle.readlines()
    with open(ligand_pdbqt, "r", encoding="utf-8") as handle:
        ligand_lines = handle.readlines()

    ligand_clean = [
        line for line in ligand_lines if not line.startswith(("MODEL", "ENDMDL"))
    ]

    with open(complex_pdbqt, "w", encoding="utf-8") as handle:
        handle.writelines(receptor_lines)
        if receptor_lines and not receptor_lines[-1].startswith("TER"):
            handle.write("TER\n")
        handle.writelines(ligand_clean)
        if ligand_clean and not ligand_clean[-1].startswith("END"):
            handle.write("END\n")


if __name__ == "__main__":
    df = dock(
        receptor="receptor.pdbqt",
        ligand="ligand.pdbqt",
        config="1A28_vina_config.txt",
    )
    print(df)
