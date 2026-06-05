#!/usr/bin/env python3
"""
Ligand preparation — convert SMILES or file input to PDBQT via MOL2 intermediate.

Workflow:
    1. SMILES: RDKit/OpenBabel → 3D → MOL2 → PDBQT
    2. File input (PDB/MOL/SDF/MOL2): OpenBabel → MOL2 → PDBQT

Key design decisions (based on extensive testing):
    - PDBQT MUST be converted from MOL2 (not directly from other formats)
    - All file inputs go through MOL2 intermediate step
    - OpenBabel uses Python import (pybel), not command-line subprocess
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import argparse
import csv
import os
import re
import signal
import sys
import multiprocessing
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
except ImportError:
    Chem = None
    AllChem = None
    RDLogger = None

try:
    from openbabel import openbabel as ob
    from openbabel import pybel
except ImportError:
    ob = None
    pybel = None


def _check_deps(mode: str = "rdkit", batch_mode: bool = False) -> None:
    """Check that required dependencies are installed.
    
    Args:
        mode: Either "rdkit" (requires RDKit) or "openbabel" (requires OpenBabel).
        batch_mode: If True, also check for tqdm.
    
    Raises:
        ImportError: If the required library is not installed.
    """
    if mode == "rdkit" and Chem is None:
        raise ImportError(
            "RDKit is required but not installed. "
            "Install with: pip install rdkit"
        )
    if mode == "openbabel" and (ob is None or pybel is None):
        raise ImportError(
            "OpenBabel is required but not installed. "
            "Install with: conda install -c conda-forge openbabel"
        )
    if batch_mode and tqdm is None:
        raise ImportError(
            "tqdm is required for batch mode but not installed. "
            "Install with: pip install tqdm"
        )


def _file_to_mol2_openbabel(
    input_path: str, 
    mol2_path: str
) -> bool:
    """Convert file (PDB/MOL/SDF/MOL2) to MOL2 using OpenBabel.
    
    All file inputs must go through MOL2 intermediate before PDBQT conversion.
    
    Args:
        input_path: Input file path (.pdb, .mol, .sdf, .mol2).
        mol2_path: Output MOL2 file path.
    
    Returns:
        True if successful, False otherwise.
    
    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If file format is unsupported.
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    ext = Path(input_path).suffix.lower()
    format_map = {
        '.pdb': 'pdb',
        '.mol': 'mol',
        '.sdf': 'sdf',
        '.mol2': 'mol2',
    }
    
    if ext not in format_map:
        raise ValueError(
            f"Unsupported file format: {ext}. "
            f"Supported: {', '.join(format_map.keys())}"
        )
    
    try:
        input_format = format_map[ext]
        mol = next(pybel.readfile(input_format, input_path), None)
        if mol is None:
            return False
        
        # Add hydrogens if not present
        mol.addh()
        
        # Write to MOL2
        mol.write("mol2", mol2_path, overwrite=True)
        return True
    except Exception:
        return False


def _smiles_to_sdf_rdkit(
    smiles: str, 
    sdf_path: str, 
    seed: int | None = None,
    timeout: int = 30
) -> bool:
    """Convert SMILES to SDF using RDKit ETKDG + MMFF optimization.
    
    Args:
        smiles: SMILES string.
        sdf_path: Output SDF file path.
        seed: Random seed for 3D generation. None for random.
        timeout: Maximum time in seconds for RDKit optimization. Default 30s.
    
    Returns:
        True if successful, False otherwise.
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"RDKit optimization exceeded {timeout}s timeout")
    
    try:
        # Set timeout signal (only works on Unix)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # Add hydrogens (required for ETKDG)
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        embed_seed = seed if seed is not None else -1
        if AllChem.EmbedMolecule(mol, randomSeed=embed_seed) == -1:
            return False
        
        # Optimize geometry with MMFF
        # If MMFF fails, return False to fallback to OpenBabel
        try:
            result = AllChem.MMFFOptimizeMolecule(mol)
            if result == -1:
                return False
        except Exception:
            return False
        
        # Write to SDF
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        
        return True
    except TimeoutError:
        return False
    except Exception:
        return False
    finally:
        # Cancel the alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _sdf_to_mol2_openbabel(sdf_path: str, mol2_path: str) -> bool:
    """Convert SDF to MOL2 using OpenBabel (import mode).
    
    Args:
        sdf_path: Input SDF file path.
        mol2_path: Output MOL2 file path.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        mol = next(pybel.readfile("sdf", sdf_path), None)
        if mol is None:
            return False
        
        mol.write("mol2", mol2_path, overwrite=True)
        return True
    except Exception:
        return False


def _gen3d_worker(smiles: str, mol2_path: str, result_queue: multiprocessing.Queue) -> None:
    """Worker function for gen3D in separate process.
    
    This is needed because signal.SIGALRM cannot interrupt C++ extensions.
    """
    try:
        from openbabel import openbabel as ob
        
        ob_conversion = ob.OBConversion()
        ob_conversion.SetInAndOutFormats("smi", "mol2")
        
        mol = ob.OBMol()
        if not ob_conversion.ReadString(mol, smiles):
            result_queue.put(False)
            return
        
        mol.AddHydrogens()
        
        gen3d = ob.OBOp.FindType("gen3D")
        if gen3d is None:
            result_queue.put(False)
            return
        
        if not gen3d.Do(mol, "--best"):
            result_queue.put(False)
            return
        
        if not ob_conversion.WriteFile(mol, mol2_path):
            result_queue.put(False)
            return
        
        result_queue.put(True)
    except Exception:
        result_queue.put(False)


def _smiles_to_mol2_openbabel(
    smiles: str, 
    mol2_path: str,
    timeout: int = 10
) -> bool:
    """Convert SMILES to 3D MOL2 using OpenBabel (import mode) as fallback.
    
    Workflow: SMILES → 2D → 3D (gen3D with --best) → MOL2
    Uses OBOp.gen3D with --best parameter for more thorough conformation search.
    
    Uses multiprocessing to enforce timeout since signal.SIGALRM cannot
    interrupt C++ extensions like gen3d.Do().
    
    Args:
        smiles: SMILES string.
        mol2_path: Output MOL2 file path.
        timeout: Maximum time in seconds for OpenBabel gen3D. Default 10s.
    
    Returns:
        True if successful, False otherwise.
    """
    try:
        # Use multiprocessing to enforce timeout
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_gen3d_worker,
            args=(smiles, mol2_path, result_queue)
        )
        process.start()
        process.join(timeout)
        
        if process.is_alive():
            # Timeout - kill the process
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join()
            return False
        
        # Get result from queue
        try:
            return result_queue.get_nowait()
        except Exception:
            return False
    except Exception:
        return False


def _mol2_to_pdbqt_openbabel(
    mol2_path: str, 
    pdbqt_path: str,
    remove_hydrogens: bool = False
) -> bool:
    """Convert MOL2 to PDBQT using OpenBabel (import mode).
    
    This is the tested and most reliable conversion path for PDBQT generation.
    
    Args:
        mol2_path: Input MOL2 file path.
        pdbqt_path: Output PDBQT file path.
        remove_hydrogens: If True, remove hydrogen atoms in PDBQT. 
            If False (default), keep hydrogens.
    
    Returns:
        True if successful, False otherwise.
    
    Raises:
        FileNotFoundError: If MOL2 file does not exist.
    """
    if not Path(mol2_path).exists():
        raise FileNotFoundError(f"MOL2 file not found: {mol2_path}")
    
    try:
        mol = next(pybel.readfile("mol2", mol2_path), None)
        if mol is None:
            return False
        
        # Compute Gasteiger charges (required for PDBQT)
        charge_model = ob.OBChargeModel.FindType("gasteiger")
        if charge_model is not None:
            charge_model.ComputeCharges(mol.OBMol)
        
        # Remove hydrogens if requested
        if remove_hydrogens:
            mol.removeh()
        
        # Write to PDBQT
        pdbqt_string = mol.write("pdbqt")
        
        os.makedirs(os.path.dirname(pdbqt_path) or ".", exist_ok=True)
        with open(pdbqt_path, "w") as f:
            f.write(pdbqt_string)
        
        return True
    except Exception:
        return False


def _prepare_ligand_from_file(
    input_path: str,
    pdbqt_path: str,
    remove_hydrogens: bool = False,
) -> tuple[str, str]:
    """Convert file input to PDBQT via MOL2 intermediate.
    
    All file formats (PDB/MOL/SDF/MOL2) are converted to MOL2 first,
    then MOL2 is converted to PDBQT.
    
    Args:
        input_path: Input file path (.pdb, .mol, .sdf, .mol2).
        pdbqt_path: Output PDBQT file path.
        remove_hydrogens: If True, remove hydrogen atoms in PDBQT.
    
    Returns:
        Tuple of (pdbqt_path, method_used) where method_used is "file_input".
    
    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If file format is unsupported.
        RuntimeError: If conversion fails.
    """
    # Create output directory and temp file path
    os.makedirs(os.path.dirname(pdbqt_path) or ".", exist_ok=True)
    tmp_dir = os.path.dirname(pdbqt_path) or "."
    base_name = Path(pdbqt_path).stem
    mol2_tmp = os.path.join(tmp_dir, f"{base_name}_tmp.mol2")
    
    try:
        if _file_to_mol2_openbabel(input_path, mol2_tmp):
            if _mol2_to_pdbqt_openbabel(mol2_tmp, pdbqt_path, remove_hydrogens=remove_hydrogens):
                return pdbqt_path, "file_input"
        
        raise RuntimeError(
            f"Failed to convert {input_path} to PDBQT."
        )
    finally:
        if os.path.exists(mol2_tmp):
            os.remove(mol2_tmp)


def _prepare_ligand_from_smiles(
    smiles: str,
    pdbqt_path: str,
    seed: int | None = None,
    remove_hydrogens: bool = False,
) -> tuple[str, str]:
    """Convert SMILES to PDBQT with RDKit priority and OpenBabel fallback.
    
    Args:
        smiles: SMILES string.
        pdbqt_path: Output PDBQT file path.
        seed: Random seed for 3D generation. None for random.
            Only used for RDKit (OpenBabel gen3D doesn't support seed).
        remove_hydrogens: If True, remove hydrogen atoms in PDBQT.
    
    Returns:
        Tuple of (pdbqt_path, method_used) where method_used is "rdkit" or "openbabel".
    
    Raises:
        RuntimeError: If all conversion methods fail.
    """
    # Create output directory and temp file paths
    os.makedirs(os.path.dirname(pdbqt_path) or ".", exist_ok=True)
    tmp_dir = os.path.dirname(pdbqt_path) or "."
    base_name = Path(pdbqt_path).stem
    
    # Suppress RDKit warnings
    if RDLogger is not None:
        RDLogger.logger().setLevel(RDLogger.CRITICAL)
    
    sdf_tmp = os.path.join(tmp_dir, f"{base_name}_tmp.sdf")
    mol2_tmp = os.path.join(tmp_dir, f"{base_name}_tmp.mol2")
    
    try:
        # Try RDKit first (ETKDG + MMFF)
        if _smiles_to_sdf_rdkit(smiles, sdf_tmp, seed=seed, timeout=30):
            if _sdf_to_mol2_openbabel(sdf_tmp, mol2_tmp):
                if _mol2_to_pdbqt_openbabel(mol2_tmp, pdbqt_path, remove_hydrogens=remove_hydrogens):
                    return pdbqt_path, "rdkit"
        
        # Fallback to OpenBabel (gen3D with --best)
        if _smiles_to_mol2_openbabel(smiles, mol2_tmp, timeout=10):
            if _mol2_to_pdbqt_openbabel(mol2_tmp, pdbqt_path, remove_hydrogens=remove_hydrogens):
                return pdbqt_path, "openbabel"
        
        raise RuntimeError(
            f"Failed to convert SMILES to PDBQT. "
            f"Both RDKit and OpenBabel methods failed for: {smiles}"
        )
    finally:
        for tmp_file in [sdf_tmp, mol2_tmp]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)


def _load_smiles_csv(csv_path: str) -> list[tuple[str, str]]:
    """Load SMILES entries from CSV with auto-detected ID/SMILES columns.
    
    Searches for columns named 'id' or 'smiles' (case-insensitive).
    Duplicate IDs get an automatic ``_2``, ``_3`` ... suffix.
    
    Args:
        csv_path: Path to CSV file.
    
    Returns:
        List of (smiles, sanitized_id) tuples.
    
    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If required columns are missing or file is empty.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV file is empty or has no header: {csv_path}")
        
        # Case-insensitive column lookup
        col_map = {name.lower().strip(): name for name in reader.fieldnames}
        
        smiles_col = col_map.get("smiles")
        if smiles_col is None:
            raise ValueError(
                f"CSV must contain a 'SMILES' column (case-insensitive). "
                f"Found columns: {list(reader.fieldnames)}"
            )
        
        id_col = col_map.get("id")
        if id_col is None:
            raise ValueError(
                f"CSV must contain an 'ID' column (case-insensitive). "
                f"Found columns: {list(reader.fieldnames)}"
            )
        
        entries = []
        id_counts: dict[str, int] = {}
        for row_num, row in enumerate(reader, 1):
            smi = (row[smiles_col] or "").strip()
            raw_id = (row[id_col] or f"row_{row_num}").strip()
            if not smi:
                continue
            # Sanitize characters unsafe for filenames
            safe_id = re.sub(r'[\\/:*?"<>|]', "_", raw_id)
            # Deduplicate repeated IDs
            count = id_counts.get(safe_id, 0) + 1
            id_counts[safe_id] = count
            if count > 1:
                safe_id = f"{safe_id}_{count}"
            entries.append((smi, safe_id))
    
    if not entries:
        raise ValueError(f"No valid SMILES entries found in CSV: {csv_path}")
    
    return entries


def prepare_ligand_batch(
    csv_path: str,
    output_dir: str,
    seed: int | None = None,
    remove_hydrogens: bool = False,
) -> tuple[str, int, int]:
    """Batch convert SMILES from CSV to PDBQT files.
    
    Reads a CSV with ``ID`` and ``SMILES`` columns, converts each molecule,
    and writes a ``batch_summary.csv`` report into *output_dir*.
    
    Args:
        csv_path: Path to CSV file with ID and SMILES columns.
        output_dir: Directory to write PDBQT files and summary CSV.
        seed: Base random seed for 3D coordinate generation.
        remove_hydrogens: If True, remove hydrogen atoms in PDBQT output.
    
    Returns:
        Tuple of (summary_csv_path, success_count, failure_count).
    
    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If CSV format is invalid.
        ImportError: If required dependencies are missing.
    """
    _check_deps("rdkit", batch_mode=True)
    _check_deps("openbabel")
    
    entries = _load_smiles_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "batch_summary.csv")
    
    success_count = 0
    failure_count = 0
    
    # Suppress RDKit warnings during batch processing
    if RDLogger is not None:
        RDLogger.logger().setLevel(RDLogger.CRITICAL)
    
    with open(summary_path, "w", newline="", encoding="utf-8") as sf:
        writer = csv.writer(sf)
        writer.writerow(["id", "smiles", "status", "output_file", "error"])
        
        for idx, (smi, mol_id) in enumerate(
            tqdm(entries, desc="Converting", unit="mol")
        ):
            pdbqt_path = os.path.join(output_dir, f"{mol_id}.pdbqt")
            mol_seed = (seed + idx) if seed is not None else None
            try:
                _prepare_ligand_from_smiles(
                    smi, pdbqt_path, seed=mol_seed, remove_hydrogens=remove_hydrogens
                )
                writer.writerow([mol_id, smi, "success", pdbqt_path, ""])
                success_count += 1
            except Exception as e:
                writer.writerow([mol_id, smi, "failed", "", str(e)])
                failure_count += 1
    
    # Restore RDKit log level
    if RDLogger is not None:
        RDLogger.logger().setLevel(RDLogger.WARNING)
    
    return summary_path, success_count, failure_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_ligand(
    smiles: str | None = None,
    input_file: str | None = None,
    output_path: str = "ligand.pdbqt",
    seed: int | None = None,
    remove_hydrogens: bool = False,
) -> tuple[str, str]:
    """Prepare ligand PDBQT from SMILES string or file input.
    
    Uses RDKit (ETKDG + MMFF) as primary method with OpenBabel fallback for SMILES.
    File inputs (PDB/MOL/SDF/MOL2) are converted via MOL2 intermediate.
    PDBQT is always generated via MOL2 intermediate (tested as most reliable).
    
    Args:
        smiles: SMILES string. Mutually exclusive with *input_file*.
        input_file: Path to file (.pdb, .mol, .sdf, .mol2). Mutually exclusive with *smiles*.
        output_path: Output PDBQT file path.
        seed: Random seed for 3D coordinate generation. None for random.
            Only used for SMILES input.
        remove_hydrogens: If True, remove hydrogen atoms in PDBQT output. 
            Default False (hydrogens kept in output).
    
    Returns:
        Tuple of (pdbqt_path, method_used) where method_used is "rdkit", "openbabel", or "file_input".
    
    Raises:
        ValueError: If neither or both inputs are provided.
        ImportError: If required dependencies are missing.
        RuntimeError: If conversion fails.
    """
    if smiles is not None and input_file is not None:
        raise ValueError("Provide either --smiles or --input, not both.")
    if smiles is None and input_file is None:
        raise ValueError("Provide either --smiles or --input.")
    
    # Check dependencies
    _check_deps("rdkit")
    _check_deps("openbabel")
    
    if input_file is not None:
        return _prepare_ligand_from_file(
            input_file, 
            output_path, 
            remove_hydrogens=remove_hydrogens
        )
    else:
        return _prepare_ligand_from_smiles(
            smiles, 
            output_path, 
            seed=seed, 
            remove_hydrogens=remove_hydrogens
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert SMILES or file input to PDBQT via MOL2 intermediate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # SMILES conversion
  torchdock prepare-ligand --smiles "c1ccccc1" -o benzene.pdbqt

  # File input (PDB/MOL/SDF/MOL2)
  torchdock prepare-ligand -i ligand.sdf -o ligand.pdbqt
  torchdock prepare-ligand -i ligand.mol2 -o ligand.pdbqt

  # Batch conversion from CSV
  torchdock prepare-ligand -b ligands.csv -o ./pdbqt_output

  # Batch with fixed seed and remove hydrogens
  torchdock prepare-ligand -b ligands.csv -o ./pdbqt_output -s 42 -d
""",
    )
    
    # Mutually exclusive input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-smi", "--smiles",
        type=str,
        help="SMILES string (single mode).",
    )
    input_group.add_argument(
        "-i", "--input",
        type=str,
        metavar="FILE",
        help="Input file (.pdb, .mol, .sdf, .mol2).",
    )
    input_group.add_argument(
        "-b", "--batch",
        type=str,
        metavar="CSV_FILE",
        help="CSV file with ID and SMILES columns for batch conversion.",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output PDBQT file (single mode) or output directory (batch mode).",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for 3D coordinate generation. "
             "Default: random seed.",
    )
    parser.add_argument(
        "-d", "--remove-h",
        action="store_true",
        default=False,
        dest="remove_h",
        help="Remove hydrogen atoms in PDBQT output. "
             "Default: hydrogens kept in output.",
    )
    return parser


def main(args: argparse.Namespace | None = None) -> str:
    """CLI entry point."""
    parser = create_parser()
    if args is None:
        args = parser.parse_args()
    
    try:
        if args.batch is not None:
            # Batch mode
            summary_path, ok, fail = prepare_ligand_batch(
                csv_path=args.batch,
                output_dir=args.output,
                seed=args.seed,
                remove_hydrogens=args.remove_h,
            )
            print(
                f"Batch complete: {ok} succeeded, {fail} failed. "
                f"Summary: {summary_path}"
            )
            return summary_path
        else:
            # Single mode (SMILES or file)
            pdbqt_path, method = prepare_ligand(
                smiles=args.smiles,
                input_file=args.input,
                output_path=args.output,
                seed=args.seed,
                remove_hydrogens=args.remove_h,
            )
            h_status = "no H" if args.remove_h else "with H"
            print(
                f"Successfully generated PDBQT: {pdbqt_path} "
                f"(method: {method}, {h_status})"
            )
            return pdbqt_path
    except (ImportError, ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
