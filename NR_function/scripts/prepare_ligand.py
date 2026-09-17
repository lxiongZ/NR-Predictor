import argparse
import contextlib
import hashlib
import io
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger


warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
RDLogger.DisableLog("rdApp.*")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)


DEFAULT_DATA_FOLDER = "ago_ant/random"
DEFAULT_OUTPUT_DIR = "ligand_cache"
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class LigandJob:
    ligand_id: str
    canonical_smiles: str
    output_pdbqt: str
    overwrite: bool
    add_hydrogens: bool
    random_seed: int


def canonicalize_smiles(smiles: str) -> Tuple[Optional[str], Optional[str]]:
    """Return canonical SMILES and an optional error message."""
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None, "invalid_smiles"
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), None
    except Exception as exc:
        return None, str(exc)


def ligand_id_from_canonical_smiles(canonical_smiles: str, digest_size: int = 16) -> str:
    digest = hashlib.sha1(canonical_smiles.encode("utf-8")).hexdigest()[:digest_size]
    return f"lig_{digest}"


def _optimize_molecule(mol: Chem.Mol) -> None:
    """Optimize with MMFF when possible, otherwise fall back to UFF."""
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        # Some molecules still produce a usable conformer even if optimization fails.
        pass


def smiles_to_pdbqt(
    smiles: str,
    output_pdbqt: str,
    add_hydrogens: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
    verbose: bool = True,
) -> str:
    """Convert one SMILES string to a ligand PDBQT file."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    output_path = Path(output_pdbqt)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Processing SMILES: {smiles}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    if add_hydrogens:
        mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status == -1:
        embed_status = AllChem.EmbedMolecule(
            mol,
            randomSeed=int(random_seed),
            useRandomCoords=True,
        )
    if embed_status == -1:
        raise ValueError(f"Failed to generate 3D conformer for: {smiles}")

    _optimize_molecule(mol)

    preparator = MoleculePreparation()
    if verbose:
        mol_setups = preparator.prepare(mol)
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
    else:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            mol_setups = preparator.prepare(mol)
            if not mol_setups:
                raise ValueError("Meeko did not return a molecule setup")
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])

    if not mol_setups:
        raise ValueError("Meeko did not return a molecule setup")
    if not is_ok:
        raise ValueError(f"Failed to write PDBQT: {error_msg}")

    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_output.write_text(pdbqt_string, encoding="utf-8")
    os.replace(temp_output, output_path)

    if verbose:
        print(f"Ligand PDBQT saved to: {output_path}")
    return str(output_path)


def _prepare_ligand_job(job: LigandJob) -> Dict[str, str]:
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    RDLogger.DisableLog("rdApp.*")

    output_path = Path(job.output_pdbqt)
    if output_path.exists() and output_path.stat().st_size > 0 and not job.overwrite:
        return {
            "ligand_id": job.ligand_id,
            "canonical_smiles": job.canonical_smiles,
            "pdbqt_path": str(output_path),
            "status": "skipped_existing",
            "error": "",
        }

    try:
        smiles_to_pdbqt(
            job.canonical_smiles,
            str(output_path),
            add_hydrogens=job.add_hydrogens,
            random_seed=job.random_seed,
            verbose=False,
        )
        return {
            "ligand_id": job.ligand_id,
            "canonical_smiles": job.canonical_smiles,
            "pdbqt_path": str(output_path),
            "status": "prepared",
            "error": "",
        }
    except Exception as exc:
        if output_path.exists() and output_path.stat().st_size == 0:
            output_path.unlink()
        return {
            "ligand_id": job.ligand_id,
            "canonical_smiles": job.canonical_smiles,
            "pdbqt_path": str(output_path),
            "status": "failed",
            "error": str(exc),
        }


def _read_dataset_rows(data_folder: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for csv_path in sorted(Path(data_folder).glob("*.csv")):
        df = pd.read_csv(csv_path)
        if "SMILES" not in df.columns:
            raise ValueError(f"Missing SMILES column: {csv_path}")
        if "Label" not in df.columns:
            raise ValueError(f"Missing Label column: {csv_path}")

        for row_index, row in df.reset_index(drop=True).iterrows():
            smiles = str(row["SMILES"])
            canonical_smiles, error = canonicalize_smiles(smiles)
            ligand_id = (
                ligand_id_from_canonical_smiles(canonical_smiles)
                if canonical_smiles is not None
                else ""
            )
            rows.append(
                {
                    "target": csv_path.stem,
                    "row_index": int(row_index),
                    "SMILES": smiles,
                    "Label": row["Label"],
                    "canonical_smiles": canonical_smiles or "",
                    "ligand_id": ligand_id,
                    "canonical_status": "ok" if canonical_smiles is not None else "failed",
                    "canonical_error": error or "",
                }
            )

    if not rows:
        raise ValueError(f"No CSV files found in {data_folder}")
    return pd.DataFrame(rows)


def _iter_jobs(
    unique_ligands: pd.DataFrame,
    ligand_dir: Path,
    overwrite: bool,
    add_hydrogens: bool,
    random_seed: int,
) -> Iterable[LigandJob]:
    for row in unique_ligands.itertuples(index=False):
        pdbqt_path = ligand_dir / f"{row.ligand_id}.pdbqt"
        yield LigandJob(
            ligand_id=row.ligand_id,
            canonical_smiles=row.canonical_smiles,
            output_pdbqt=str(pdbqt_path),
            overwrite=overwrite,
            add_hydrogens=add_hydrogens,
            random_seed=random_seed,
        )


def prepare_ligand_cache(
    data_folder: str = DEFAULT_DATA_FOLDER,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    workers: Optional[int] = None,
    overwrite: bool = False,
    add_hydrogens: bool = True,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Prepare a global ligand PDBQT cache for all ago_ant CSV files."""
    output_path = Path(output_dir)
    ligand_dir = output_path / "pdbqt"
    output_path.mkdir(parents=True, exist_ok=True)
    ligand_dir.mkdir(parents=True, exist_ok=True)

    row_manifest = _read_dataset_rows(data_folder)
    valid_rows = row_manifest[row_manifest["canonical_status"] == "ok"].copy()
    unique_ligands = (
        valid_rows[["ligand_id", "canonical_smiles"]]
        .drop_duplicates("ligand_id")
        .sort_values("ligand_id")
        .reset_index(drop=True)
    )

    jobs = list(
        _iter_jobs(
            unique_ligands,
            ligand_dir=ligand_dir,
            overwrite=overwrite,
            add_hydrogens=add_hydrogens,
            random_seed=random_seed,
        )
    )
    print(f"Rows: {len(row_manifest)}")
    print(f"Unique valid ligands: {len(jobs)}")
    print(f"Workers: {workers or os.cpu_count() or 1}")

    results: List[Dict[str, str]] = []
    if workers == 1:
        for idx, job in enumerate(jobs, 1):
            results.append(_prepare_ligand_job(job))
            if idx % 100 == 0 or idx == len(jobs):
                print(f"[{idx}/{len(jobs)}] ligand preparation checked")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_job = {executor.submit(_prepare_ligand_job, job): job for job in jobs}
            for idx, future in enumerate(as_completed(future_to_job), 1):
                results.append(future.result())
                if idx % 100 == 0 or idx == len(jobs):
                    print(f"[{idx}/{len(jobs)}] ligand preparation checked")

    result_df = pd.DataFrame(results)
    if result_df.empty:
        result_df = pd.DataFrame(
            columns=["ligand_id", "canonical_smiles", "pdbqt_path", "status", "error"]
        )

    row_manifest = row_manifest.merge(
        result_df[["ligand_id", "pdbqt_path", "status", "error"]],
        on="ligand_id",
        how="left",
    )
    invalid_mask = row_manifest["canonical_status"] != "ok"
    row_manifest.loc[invalid_mask, "status"] = "failed"
    row_manifest.loc[invalid_mask, "error"] = row_manifest.loc[invalid_mask, "canonical_error"]
    row_manifest["pdbqt_path"] = row_manifest["pdbqt_path"].fillna("")
    row_manifest["status"] = row_manifest["status"].fillna("failed")
    row_manifest["error"] = row_manifest["error"].fillna("")

    row_manifest_file = output_path / "ligand_manifest.csv"
    unique_manifest_file = output_path / "unique_ligands.csv"
    row_manifest.to_csv(row_manifest_file, index=False)
    result_df.to_csv(unique_manifest_file, index=False)

    status_counts = row_manifest["status"].value_counts(dropna=False).to_dict()
    print(f"Saved row manifest: {row_manifest_file}")
    print(f"Saved unique ligand manifest: {unique_manifest_file}")
    print(f"Row status counts: {status_counts}")
    return row_manifest


def batch_prepare_ligands(smiles_file: str, output_dir: str) -> None:
    """Backward-compatible plain-text SMILES batch conversion."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(smiles_file, "r", encoding="utf-8") as handle:
        for i, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=1)
            smiles = parts[0]
            name = parts[1] if len(parts) > 1 else f"ligand_{i}"
            output_pdbqt = os.path.join(output_dir, f"{name}.pdbqt")

            try:
                smiles_to_pdbqt(smiles, output_pdbqt)
                print(f"OK: {name}")
            except Exception as exc:
                print(f"FAILED: {name}: {exc}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ligand PDBQT files")
    subparsers = parser.add_subparsers(dest="command")

    cache = subparsers.add_parser("cache", help="Prepare global ligand cache from target CSV files, defaulting to ago_ant/random")
    cache.add_argument("--data_folder", default=DEFAULT_DATA_FOLDER)
    cache.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    cache.add_argument("--workers", type=int, default=None)
    cache.add_argument("--overwrite", action="store_true")
    cache.add_argument("--no_hydrogens", action="store_true")
    cache.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED)

    one = subparsers.add_parser("one", help="Prepare one ligand from SMILES")
    one.add_argument("--smiles", required=True)
    one.add_argument("--output", required=True)
    one.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED)
    one.add_argument("--no_hydrogens", action="store_true")

    batch = subparsers.add_parser("batch", help="Prepare ligands from a text file")
    batch.add_argument("--smiles_file", required=True)
    batch.add_argument("--output", required=True)

    parser.add_argument("--smiles", help=argparse.SUPPRESS)
    parser.add_argument("--smiles_file", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    parser = build_arg_parser()
    args = parser.parse_args()

    # Backward compatibility with the previous CLI:
    #   python prepare_ligand.py --smiles CCO --output out.pdbqt
    #   python prepare_ligand.py --smiles_file input.smi --output out_dir
    if args.command is None and args.smiles:
        if not args.output:
            parser.error("--output is required with --smiles")
        smiles_to_pdbqt(args.smiles, args.output)
        return
    if args.command is None and args.smiles_file:
        if not args.output:
            parser.error("--output is required with --smiles_file")
        batch_prepare_ligands(args.smiles_file, args.output)
        return

    if args.command == "cache":
        prepare_ligand_cache(
            data_folder=args.data_folder,
            output_dir=args.output_dir,
            workers=args.workers,
            overwrite=args.overwrite,
            add_hydrogens=not args.no_hydrogens,
            random_seed=args.random_seed,
        )
    elif args.command == "one":
        smiles_to_pdbqt(
            args.smiles,
            args.output,
            add_hydrogens=not args.no_hydrogens,
            random_seed=args.random_seed,
        )
    elif args.command == "batch":
        batch_prepare_ligands(args.smiles_file, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
