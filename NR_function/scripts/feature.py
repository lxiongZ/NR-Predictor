import argparse
import csv
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from prepare_ligand import DEFAULT_OUTPUT_DIR, prepare_ligand_cache
from vina.vina_docking_wrapper import dock_one


warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
RDLogger.DisableLog("rdApp.*")


EXCLUDED_RDKIT_DESCRIPTORS = {
    "Ipc",
    "SMR_VSA8",
    "SlogP_VSA9",
    "fr_diazo",
    "fr_prisulfonamd",
}
RDKIT_DESCRIPTOR_LIST = [
    (name, desc)
    for name, desc in Descriptors.descList
    if name not in EXCLUDED_RDKIT_DESCRIPTORS
]

DEFAULT_DATA_FOLDER = "ago_ant/random"
DEFAULT_PDB_ROOT = "PDB"
DEFAULT_DOCKING_SCORE_DIR = "docking_scores"
DEFAULT_FEATURE_OUTPUT_DIR = "ago_ant_features/processed_features"
DEFAULT_VINA_EXE = "vina/vina_1.2.5_linux_x86_64"
RECEPTOR_KINDS = ("Agonist", "Antagonist")
SUCCESS_LIGAND_STATUSES = {"prepared", "skipped_existing"}
SCORE_COLUMNS = [
    "target",
    "receptor_kind",
    "receptor_id",
    "ligand_id",
    "affinity",
    "status",
    "error",
    "receptor_path",
    "config_path",
    "pdbqt_path",
]


@dataclass(frozen=True)
class ReceptorEntry:
    target: str
    receptor_kind: str
    receptor_id: str
    receptor_path: str
    config_path: str


@dataclass(frozen=True)
class DockingJob:
    target: str
    receptor_kind: str
    receptor_id: str
    receptor_path: str
    config_path: str
    ligand_id: str
    pdbqt_path: str
    score_file: str


def smiles_to_rdkit_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(len(RDKIT_DESCRIPTOR_LIST))

    try:
        desc_values = [desc(mol) for _, desc in RDKIT_DESCRIPTOR_LIST]
        descriptors = np.array(desc_values, dtype=float)
        return np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as exc:
        print(f"Error calculating descriptors for {smiles}: {exc}")
        return np.zeros(len(RDKIT_DESCRIPTOR_LIST))


def receptor_id_from_path(receptor_path: Path) -> str:
    stem = receptor_path.stem
    if stem.endswith("_clean"):
        return stem[: -len("_clean")]
    return stem


def find_config_for_receptor(receptor_path: Path, receptor_dir: Path) -> Path:
    configs = sorted(receptor_dir.glob("**/*vina_config*.txt"))
    if not configs:
        raise FileNotFoundError(f"No Vina config found under {receptor_dir}")

    stem = receptor_path.stem
    base = receptor_id_from_path(receptor_path)

    def score(config_path: Path) -> int:
        name = config_path.name
        if name.startswith(f"{stem}_vina_config"):
            return 0
        if stem in name:
            return 1
        if name.startswith(f"{base}_vina_config"):
            return 2
        if base in name and "vina_config" in name:
            return 3
        return 100

    ranked = sorted((score(path), path) for path in configs)
    best_score, best_path = ranked[0]
    if best_score >= 100:
        raise FileNotFoundError(
            f"No matching Vina config for {receptor_path} under {receptor_dir}"
        )
    return best_path


def discover_receptors(
    pdb_root: str = DEFAULT_PDB_ROOT,
    targets: Optional[Sequence[str]] = None,
    receptor_kinds: Sequence[str] = RECEPTOR_KINDS,
) -> pd.DataFrame:
    target_filter = set(targets or [])
    entries: List[ReceptorEntry] = []
    missing: List[str] = []

    for target_dir in sorted(Path(pdb_root).iterdir()):
        if not target_dir.is_dir():
            continue
        target = target_dir.name
        if target_filter and target not in target_filter:
            continue

        for receptor_kind in receptor_kinds:
            receptor_dir = target_dir / receptor_kind
            if not receptor_dir.exists():
                continue

            for receptor_path in sorted(receptor_dir.glob("*_clean.pdbqt")):
                receptor_id = receptor_id_from_path(receptor_path)
                try:
                    config_path = find_config_for_receptor(receptor_path, receptor_dir)
                except FileNotFoundError as exc:
                    missing.append(str(exc))
                    continue
                entries.append(
                    ReceptorEntry(
                        target=target,
                        receptor_kind=receptor_kind,
                        receptor_id=receptor_id,
                        receptor_path=str(receptor_path),
                        config_path=str(config_path),
                    )
                )

    if missing:
        print("Warning: some receptors were skipped because configs were missing:")
        for item in missing:
            print(f"  {item}")

    manifest = pd.DataFrame([entry.__dict__ for entry in entries])
    if manifest.empty:
        raise ValueError(f"No receptors discovered under {pdb_root}")

    return manifest.sort_values(["target", "receptor_kind", "receptor_id"]).reset_index(
        drop=True
    )


def score_file_for_receptor(
    score_dir: str,
    target: str,
    receptor_kind: str,
    receptor_id: str,
) -> Path:
    return Path(score_dir) / target / receptor_kind / f"{receptor_id}.csv"


def load_ligand_manifest(
    ligand_cache_dir: str = DEFAULT_OUTPUT_DIR,
    data_folder: str = DEFAULT_DATA_FOLDER,
    ligand_workers: Optional[int] = None,
    prepare_if_missing: bool = True,
) -> pd.DataFrame:
    manifest_file = Path(ligand_cache_dir) / "ligand_manifest.csv"
    if not manifest_file.exists():
        if not prepare_if_missing:
            raise FileNotFoundError(
                f"Missing ligand manifest: {manifest_file}. Run prepare_ligand.py cache first."
            )
        prepare_ligand_cache(
            data_folder=data_folder,
            output_dir=ligand_cache_dir,
            workers=ligand_workers,
        )

    manifest = pd.read_csv(manifest_file)
    required = {"target", "row_index", "SMILES", "Label", "ligand_id", "pdbqt_path", "status"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Ligand manifest is missing columns: {sorted(missing)}")
    return manifest


def load_existing_score_ids(score_file: Path, retry_failed: bool) -> set:
    if not score_file.exists() or score_file.stat().st_size == 0:
        return set()

    try:
        df = pd.read_csv(score_file)
    except pd.errors.EmptyDataError:
        return set()

    if "ligand_id" not in df.columns:
        return set()
    if retry_failed and "status" in df.columns:
        df = df[df["status"] == "success"]
    return set(df["ligand_id"].astype(str))


def make_docking_jobs(
    receptor_manifest: pd.DataFrame,
    ligand_manifest: pd.DataFrame,
    score_dir: str = DEFAULT_DOCKING_SCORE_DIR,
    retry_failed: bool = False,
) -> List[DockingJob]:
    jobs: List[DockingJob] = []
    ligand_manifest = ligand_manifest.copy()
    ligand_manifest["ligand_id"] = ligand_manifest["ligand_id"].fillna("").astype(str)
    ligand_manifest["pdbqt_path"] = ligand_manifest["pdbqt_path"].fillna("").astype(str)

    for receptor in receptor_manifest.itertuples(index=False):
        score_file = score_file_for_receptor(
            score_dir,
            receptor.target,
            receptor.receptor_kind,
            receptor.receptor_id,
        )
        existing_ids = load_existing_score_ids(score_file, retry_failed=retry_failed)

        target_ligands = ligand_manifest[
            (ligand_manifest["target"] == receptor.target)
            & (ligand_manifest["status"].isin(SUCCESS_LIGAND_STATUSES))
            & (ligand_manifest["ligand_id"] != "")
            & (ligand_manifest["pdbqt_path"] != "")
        ].copy()
        target_ligands = target_ligands.drop_duplicates("ligand_id")

        for ligand in target_ligands.itertuples(index=False):
            ligand_id = str(ligand.ligand_id)
            if ligand_id in existing_ids:
                continue
            pdbqt_path = str(ligand.pdbqt_path)
            if not Path(pdbqt_path).exists():
                continue
            jobs.append(
                DockingJob(
                    target=receptor.target,
                    receptor_kind=receptor.receptor_kind,
                    receptor_id=receptor.receptor_id,
                    receptor_path=receptor.receptor_path,
                    config_path=receptor.config_path,
                    ligand_id=ligand_id,
                    pdbqt_path=pdbqt_path,
                    score_file=str(score_file),
                )
            )

    return jobs


def append_score_row(score_file: str, row: Dict[str, object]) -> None:
    path = Path(score_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0

    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in SCORE_COLUMNS})


def run_docking_job(
    job: DockingJob,
    vina_exe: str,
    cpu_per_job: int,
    seed: int,
) -> Dict[str, object]:
    result = dock_one(
        receptor=job.receptor_path,
        ligand=job.pdbqt_path,
        config=job.config_path,
        vina_exe=vina_exe,
        cpu=cpu_per_job,
        seed=seed,
    )
    return {
        "target": job.target,
        "receptor_kind": job.receptor_kind,
        "receptor_id": job.receptor_id,
        "ligand_id": job.ligand_id,
        "affinity": result.get("affinity", float("nan")),
        "status": result.get("status", "failed"),
        "error": result.get("error", ""),
        "receptor_path": job.receptor_path,
        "config_path": job.config_path,
        "pdbqt_path": job.pdbqt_path,
        "score_file": job.score_file,
    }


def run_parallel_docking(
    data_folder: str = DEFAULT_DATA_FOLDER,
    pdb_root: str = DEFAULT_PDB_ROOT,
    ligand_cache_dir: str = DEFAULT_OUTPUT_DIR,
    score_dir: str = DEFAULT_DOCKING_SCORE_DIR,
    vina_exe: str = DEFAULT_VINA_EXE,
    targets: Optional[Sequence[str]] = None,
    receptor_kinds: Sequence[str] = RECEPTOR_KINDS,
    ligand_workers: Optional[int] = None,
    docking_workers: int = 4,
    cpu_per_job: int = 1,
    seed: int = 42,
    retry_failed: bool = False,
    progress_every: int = 100,
) -> pd.DataFrame:
    receptor_manifest = discover_receptors(
        pdb_root=pdb_root,
        targets=targets,
        receptor_kinds=receptor_kinds,
    )
    ligand_manifest = load_ligand_manifest(
        ligand_cache_dir=ligand_cache_dir,
        data_folder=data_folder,
        ligand_workers=ligand_workers,
        prepare_if_missing=True,
    )

    jobs = make_docking_jobs(
        receptor_manifest=receptor_manifest,
        ligand_manifest=ligand_manifest,
        score_dir=score_dir,
        retry_failed=retry_failed,
    )

    print(f"Receptors: {len(receptor_manifest)}")
    print(f"Pending docking jobs: {len(jobs)}")
    print(f"Docking workers: {docking_workers}, Vina CPU per job: {cpu_per_job}")

    if not jobs:
        return receptor_manifest

    completed = 0
    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=docking_workers) as executor:
        futures = [
            executor.submit(run_docking_job, job, vina_exe, cpu_per_job, seed)
            for job in jobs
        ]
        for future in as_completed(futures):
            row = future.result()
            append_score_row(str(row["score_file"]), row)
            completed += 1
            if row["status"] == "success":
                success += 1
            else:
                failed += 1

            if progress_every and (completed % progress_every == 0 or completed == len(jobs)):
                print(
                    f"[{completed}/{len(jobs)}] docking finished "
                    f"({success} success, {failed} failed)"
                )

    return receptor_manifest


def load_latest_scores(score_file: Path) -> pd.DataFrame:
    if not score_file.exists() or score_file.stat().st_size == 0:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    try:
        score_df = pd.read_csv(score_file)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    if "ligand_id" not in score_df.columns:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    return score_df.drop_duplicates("ligand_id", keep="last")


def add_affinity_summaries(feature_df: pd.DataFrame, affinity_columns: Dict[str, List[str]]) -> None:
    for receptor_kind, columns in affinity_columns.items():
        if not columns:
            continue
        prefix = f"affinity_{receptor_kind}"
        feature_df[f"{prefix}_min"] = feature_df[columns].min(axis=1, skipna=True)
        feature_df[f"{prefix}_mean"] = feature_df[columns].mean(axis=1, skipna=True)
        feature_df[f"{prefix}_max"] = feature_df[columns].max(axis=1, skipna=True)
        feature_df[f"{prefix}_success_count"] = feature_df[columns].notna().sum(axis=1)

    agonist_min = "affinity_Agonist_min"
    antagonist_min = "affinity_Antagonist_min"
    if agonist_min in feature_df.columns and antagonist_min in feature_df.columns:
        feature_df["affinity_delta_Agonist_min_minus_Antagonist_min"] = (
            feature_df[agonist_min] - feature_df[antagonist_min]
        )


def assemble_features(
    data_folder: str = DEFAULT_DATA_FOLDER,
    pdb_root: str = DEFAULT_PDB_ROOT,
    ligand_cache_dir: str = DEFAULT_OUTPUT_DIR,
    score_dir: str = DEFAULT_DOCKING_SCORE_DIR,
    output_dir: str = DEFAULT_FEATURE_OUTPUT_DIR,
    targets: Optional[Sequence[str]] = None,
    receptor_kinds: Sequence[str] = RECEPTOR_KINDS,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    receptor_manifest = discover_receptors(
        pdb_root=pdb_root,
        targets=targets,
        receptor_kinds=receptor_kinds,
    )
    ligand_manifest = load_ligand_manifest(
        ligand_cache_dir=ligand_cache_dir,
        data_folder=data_folder,
        prepare_if_missing=False,
    )

    rdkit_names = [name for name, _ in RDKIT_DESCRIPTOR_LIST]
    csv_files = sorted(Path(data_folder).glob("*.csv"))
    target_filter = set(targets or [])
    if target_filter:
        csv_files = [path for path in csv_files if path.stem in target_filter]

    for csv_path in csv_files:
        target = csv_path.stem
        print(f"Assembling features for {target}")
        source_df = pd.read_csv(csv_path).reset_index(drop=True)
        smiles_list = source_df["SMILES"].astype(str).tolist()
        labels = source_df["Label"].values

        rdkit_features = np.array([smiles_to_rdkit_descriptors(smiles) for smiles in smiles_list])
        feature_df = pd.DataFrame(rdkit_features, columns=rdkit_names)
        feature_df.insert(0, "SMILES", smiles_list)

        target_ligands = ligand_manifest[ligand_manifest["target"] == target].copy()
        target_ligands = target_ligands.sort_values("row_index")
        if len(target_ligands) != len(source_df):
            raise ValueError(
                f"Ligand manifest row count mismatch for {target}: "
                f"{len(target_ligands)} vs {len(source_df)}"
            )

        ligand_ids = target_ligands["ligand_id"].fillna("").astype(str).tolist()
        affinity_columns: Dict[str, List[str]] = {kind: [] for kind in RECEPTOR_KINDS}

        target_receptors = receptor_manifest[receptor_manifest["target"] == target]
        for receptor in target_receptors.itertuples(index=False):
            score_file = score_file_for_receptor(
                score_dir,
                receptor.target,
                receptor.receptor_kind,
                receptor.receptor_id,
            )
            score_df = load_latest_scores(score_file)
            if "status" in score_df.columns:
                successful = score_df[score_df["status"] == "success"].copy()
            else:
                successful = score_df.iloc[0:0].copy()
            affinity_map = dict(zip(successful["ligand_id"].astype(str), successful["affinity"]))
            column_name = f"affinity_{receptor.receptor_kind}_{receptor.receptor_id}"
            feature_df[column_name] = [affinity_map.get(ligand_id, np.nan) for ligand_id in ligand_ids]
            affinity_columns.setdefault(receptor.receptor_kind, []).append(column_name)

        add_affinity_summaries(feature_df, affinity_columns)
        feature_df["Label"] = labels

        output_file = output_path / f"{target}_features.csv"
        feature_df.to_csv(output_file, index=False)
        print(f"Saved {output_file} shape={feature_df.shape}")


def generate_features_with_docking(
    data_folder: str = DEFAULT_DATA_FOLDER,
    pdb_root: str = DEFAULT_PDB_ROOT,
    ligand_cache_dir: str = DEFAULT_OUTPUT_DIR,
    score_dir: str = DEFAULT_DOCKING_SCORE_DIR,
    output_dir: str = DEFAULT_FEATURE_OUTPUT_DIR,
    vina_exe: str = DEFAULT_VINA_EXE,
    ligand_workers: Optional[int] = None,
    docking_workers: int = 4,
    cpu_per_job: int = 1,
    targets: Optional[Sequence[str]] = None,
    retry_failed: bool = False,
) -> None:
    prepare_ligand_cache(
        data_folder=data_folder,
        output_dir=ligand_cache_dir,
        workers=ligand_workers,
    )
    run_parallel_docking(
        data_folder=data_folder,
        pdb_root=pdb_root,
        ligand_cache_dir=ligand_cache_dir,
        score_dir=score_dir,
        vina_exe=vina_exe,
        targets=targets,
        ligand_workers=ligand_workers,
        docking_workers=docking_workers,
        cpu_per_job=cpu_per_job,
        retry_failed=retry_failed,
    )
    assemble_features(
        data_folder=data_folder,
        pdb_root=pdb_root,
        ligand_cache_dir=ligand_cache_dir,
        score_dir=score_dir,
        output_dir=output_dir,
        targets=targets,
    )


def parse_targets(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    targets: List[str] = []
    for value in values:
        targets.extend(item.strip() for item in value.split(",") if item.strip())
    return targets or None


def parse_receptor_kinds(values: Optional[List[str]]) -> List[str]:
    if not values:
        return list(RECEPTOR_KINDS)
    kinds: List[str] = []
    for value in values:
        kinds.extend(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(kinds).difference(RECEPTOR_KINDS))
    if invalid:
        raise ValueError(f"Invalid receptor kinds: {invalid}")
    return kinds


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_folder", default=DEFAULT_DATA_FOLDER)
    parser.add_argument("--pdb_root", default=DEFAULT_PDB_ROOT)
    parser.add_argument("--ligand_cache_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score_dir", default=DEFAULT_DOCKING_SCORE_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_FEATURE_OUTPUT_DIR)
    parser.add_argument("--targets", nargs="*", help="Target names, e.g. AR THRB or AR,THRB")
    parser.add_argument(
        "--receptor_kinds",
        nargs="*",
        help="Agonist, Antagonist, or both. Default: both",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate NR docking features for the upload package layout")
    subparsers = parser.add_subparsers(dest="command")

    prepare = subparsers.add_parser("prepare-ligands", help="Prepare ligand cache only")
    prepare.add_argument("--data_folder", default=DEFAULT_DATA_FOLDER)
    prepare.add_argument("--ligand_cache_dir", default=DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--ligand_workers", type=int, default=None)
    prepare.add_argument("--overwrite_ligands", action="store_true")

    manifest = subparsers.add_parser("manifest", help="Print discovered receptor manifest")
    manifest.add_argument("--pdb_root", default=DEFAULT_PDB_ROOT)
    manifest.add_argument("--targets", nargs="*")
    manifest.add_argument("--receptor_kinds", nargs="*")
    manifest.add_argument("--output")

    dock_parser = subparsers.add_parser("dock", help="Run resumable parallel docking")
    add_common_args(dock_parser)
    dock_parser.add_argument("--vina_exe", default=DEFAULT_VINA_EXE)
    dock_parser.add_argument("--ligand_workers", type=int, default=None)
    dock_parser.add_argument("--docking_workers", type=int, default=4)
    dock_parser.add_argument("--cpu_per_job", type=int, default=1)
    dock_parser.add_argument("--seed", type=int, default=42)
    dock_parser.add_argument("--retry_failed", action="store_true")

    assemble = subparsers.add_parser("assemble", help="Assemble feature CSV files from scores")
    add_common_args(assemble)

    all_parser = subparsers.add_parser("all", help="Prepare ligands, dock, and assemble features")
    add_common_args(all_parser)
    all_parser.add_argument("--vina_exe", default=DEFAULT_VINA_EXE)
    all_parser.add_argument("--ligand_workers", type=int, default=None)
    all_parser.add_argument("--docking_workers", type=int, default=4)
    all_parser.add_argument("--cpu_per_job", type=int, default=1)
    all_parser.add_argument("--retry_failed", action="store_true")

    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    parser = build_arg_parser()
    args = parser.parse_args()

    start = time.perf_counter()
    command = args.command or "all"

    if command == "prepare-ligands":
        prepare_ligand_cache(
            data_folder=args.data_folder,
            output_dir=args.ligand_cache_dir,
            workers=args.ligand_workers,
            overwrite=args.overwrite_ligands,
        )
    elif command == "manifest":
        manifest = discover_receptors(
            pdb_root=args.pdb_root,
            targets=parse_targets(args.targets),
            receptor_kinds=parse_receptor_kinds(args.receptor_kinds),
        )
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            manifest.to_csv(args.output, index=False)
            print(f"Saved {args.output}")
        else:
            print(manifest.to_string(index=False))
    elif command == "dock":
        run_parallel_docking(
            data_folder=args.data_folder,
            pdb_root=args.pdb_root,
            ligand_cache_dir=args.ligand_cache_dir,
            score_dir=args.score_dir,
            vina_exe=args.vina_exe,
            targets=parse_targets(args.targets),
            receptor_kinds=parse_receptor_kinds(args.receptor_kinds),
            ligand_workers=args.ligand_workers,
            docking_workers=args.docking_workers,
            cpu_per_job=args.cpu_per_job,
            seed=args.seed,
            retry_failed=args.retry_failed,
        )
    elif command == "assemble":
        assemble_features(
            data_folder=args.data_folder,
            pdb_root=args.pdb_root,
            ligand_cache_dir=args.ligand_cache_dir,
            score_dir=args.score_dir,
            output_dir=args.output_dir,
            targets=parse_targets(args.targets),
            receptor_kinds=parse_receptor_kinds(args.receptor_kinds),
        )
    elif command == "all":
        generate_features_with_docking(
            data_folder=args.data_folder,
            pdb_root=args.pdb_root,
            ligand_cache_dir=args.ligand_cache_dir,
            score_dir=args.score_dir,
            output_dir=args.output_dir,
            vina_exe=args.vina_exe,
            ligand_workers=args.ligand_workers,
            docking_workers=args.docking_workers,
            cpu_per_job=args.cpu_per_job,
            targets=parse_targets(args.targets),
            retry_failed=args.retry_failed,
        )
    else:
        parser.print_help()

    elapsed = time.perf_counter() - start
    print(f"Elapsed: {timedelta(seconds=int(elapsed))}")


if __name__ == "__main__":
    main()
