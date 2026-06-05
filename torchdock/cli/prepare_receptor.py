#!/usr/bin/env python3
"""
Receptor preparation — convert PDB to PDBQT via OpenBabel.

Workflow:
    1. (Optional) Clean the protein PDB: remove non-standard residues,
       hydrogens, alternative conformations, and non-amino-acid entities.
    2. Add hydrogens via OpenBabel (PDB files often lack H atoms).
    3. Compute Gasteiger charges and generate PDBQT via OpenBabel.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import argparse
import os
import sys
import tempfile
from pathlib import Path

try:
    from openbabel import openbabel as ob
    from openbabel import pybel
except ImportError:
    ob = None
    pybel = None

try:
    from Bio.PDB import PDBParser, PDBIO, Select
    from Bio.PDB.Polypeptide import is_aa
except ImportError:
    PDBParser = None
    PDBIO = None
    Select = None
    is_aa = None

# Lazy import to avoid circular dependencies and heavy startup cost.
_restype_name_to_atom14_names = None


def _get_atom14_names() -> dict[str, list[str]]:
    """Lazy-load atom14 constants."""
    global _restype_name_to_atom14_names
    if _restype_name_to_atom14_names is None:
        from torchdock.constants.atom14_constants import (
            RESTYPE_NAME_TO_ATOM14_NAMES,
        )
        _restype_name_to_atom14_names = RESTYPE_NAME_TO_ATOM14_NAMES
    return _restype_name_to_atom14_names


def _check_deps(clean_mode: bool = False) -> None:
    """Check that required dependencies are installed."""
    missing = []
    if ob is None or pybel is None:
        missing.append("openbabel")
    if clean_mode and PDBParser is None:
        missing.append("biopython")
    if missing:
        raise ImportError(
            f"prepare-receptor requires: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


class _AminoAcidSelect(Select):
    """Biopython Select filter for standard amino-acid atoms in the atom14 set.

    Used with ``Bio.PDB.PDBIO.save()`` to retain only standard amino-acid
    residues whose atoms appear in the atom14 representation, stripping
    hydrogens, non-standard residues, and alternative conformations.
    """

    def __init__(self, allowed_aa_atoms: dict[str, list[str]] | None = None) -> None:
        self.allowed_aa_atoms = allowed_aa_atoms or {}

    def accept_residue(self, residue) -> int:
        if not is_aa(residue):
            return 0
        res_name = residue.get_resname()
        if res_name not in self.allowed_aa_atoms:
            return 0
        return 1

    def accept_atom(self, atom) -> int:
        res = atom.get_parent()
        if not is_aa(res):
            return 0

        res_name = res.get_resname()
        atom_name = atom.get_name()

        # Remove hydrogens
        if atom_name.startswith("H"):
            return 0

        # Handle alternative conformations: keep only 'A' / '1'
        alt_loc = atom.get_altloc()
        if alt_loc.strip() and alt_loc not in ("A", "1"):
            print(
                f"Warning: Altloc '{alt_loc}' found in {atom.get_full_id()}, removed.",
                file=sys.stderr,
            )
            return 0
        if alt_loc in ("A", "1"):
            atom.set_altloc(" ")

        # Keep only atoms in the atom14 representation
        if res_name in self.allowed_aa_atoms:
            return atom_name in self.allowed_aa_atoms[res_name]
        return 0


def _clean_pdb(input_pdb: str, output_pdb: str) -> str:
    """Clean a protein PDB file, retaining only standard amino-acid atoms.

    Args:
        input_pdb: Path to the input PDB file.
        output_pdb: Path to write the cleaned PDB file.

    Returns:
        Path to the cleaned PDB file.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, _AminoAcidSelect(_get_atom14_names()))
    return output_pdb


def pdb_to_pdbqt_openbabel(
    pdb_path: str,
    pdbqt_path: str,
    remove_hydrogens: bool = False,
) -> str:
    """Convert PDB to PDBQT using OpenBabel.
    
    Args:
        pdb_path: Input PDB file path.
        pdbqt_path: Output PDBQT file path.
        remove_hydrogens: If True, remove hydrogen atoms. Default False.
    
    Returns:
        Path to the generated PDBQT file.
    
    Raises:
        FileNotFoundError: If input PDB does not exist.
        RuntimeError: If conversion fails.
    """
    if not Path(pdb_path).exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    
    _check_deps()
    
    try:
        # Read PDB file
        mol = next(pybel.readfile("pdb", pdb_path), None)
        if mol is None:
            raise RuntimeError(f"Failed to read PDB file: {pdb_path}")
        
        # Add hydrogens (PDB files often lack H atoms)
        # Use AddHydrogens with polaronly=True to only add polar H atoms
        mol.addh()
        
        # Compute Gasteiger charges (required for PDBQT)
        charge_model = ob.OBChargeModel.FindType("gasteiger")
        if charge_model is not None:
            charge_model.ComputeCharges(mol.OBMol)
        
        # Remove hydrogens if requested
        if remove_hydrogens:
            mol.removeh()
        
        # Use OBConversion with 'r' option to output rigid molecule (no ROOT/BRANCH)
        ob_conversion = ob.OBConversion()
        ob_conversion.SetOutFormat("pdbqt")
        ob_conversion.AddOption("r", ob_conversion.OUTOPTIONS)  # r = rigid molecule, no branches or torsion tree
        
        pdbqt_string = ob_conversion.WriteString(mol.OBMol)
        
        # Remove the auto-generated REMARK line containing the source filename
        # (OpenBabel adds "REMARK  Name = <input_filename>" which exposes temp paths)
        pdbqt_lines = [line for line in pdbqt_string.splitlines(True)
                       if not line.startswith("REMARK  Name =")]
        pdbqt_string = "".join(pdbqt_lines)
        
        os.makedirs(os.path.dirname(pdbqt_path) or ".", exist_ok=True)
        with open(pdbqt_path, "w") as f:
            f.write(pdbqt_string)
        
        return pdbqt_path
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDB to PDBQT: {e}")


def prepare_receptor(
    input_pdb: str,
    output_path: str,
    clean: bool = True,
    remove_hydrogens: bool = False,
) -> str:
    """Prepare a receptor PDBQT from a PDB file.

    Args:
        input_pdb: Path to the input PDB file.
        output_path: Output PDBQT file path.
        clean: If True, clean the protein before conversion
            (remove non-standard residues, H atoms, alt conformations).
        remove_hydrogens: If True, remove hydrogen atoms in PDBQT output.

    Returns:
        Path to the generated PDBQT file.

    Raises:
        FileNotFoundError: If the input PDB does not exist.
        ImportError: If required dependencies are missing.
        RuntimeError: If conversion fails.
    """
    if not Path(input_pdb).exists():
        raise FileNotFoundError(f"PDB file not found: {input_pdb}")

    _check_deps(clean_mode=clean)

    pdb_to_convert = input_pdb

    if clean:
        # Write cleaned PDB to a temp file, then convert that.
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
        os.close(tmp_fd)
        try:
            _clean_pdb(input_pdb, tmp_path)
            pdb_to_convert = tmp_path
            print(f"Protein cleaned: {input_pdb} -> temp cleaned PDB", file=sys.stderr)
            result = pdb_to_pdbqt_openbabel(
                pdb_path=pdb_to_convert,
                pdbqt_path=output_path,
                remove_hydrogens=remove_hydrogens,
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        result = pdb_to_pdbqt_openbabel(
            pdb_path=pdb_to_convert,
            pdbqt_path=output_path,
            remove_hydrogens=remove_hydrogens,
        )

    return result


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare receptor PDBQT from a PDB file using OpenBabel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Clean protein and convert (default)
  torchdock prepare-receptor -i protein.pdb -o receptor.pdbqt

  # Convert without cleaning
  torchdock prepare-receptor -i protein.pdb -o receptor.pdbqt -nc

  # Clean and convert, remove hydrogens
  torchdock prepare-receptor -i protein.pdb -o receptor.pdbqt -d
""",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input PDB file.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output PDBQT file.",
    )
    parser.add_argument(
        "-d", "--remove-h",
        action="store_true",
        default=False,
        help="Remove hydrogen atoms in PDBQT output. Default: keep hydrogens.",
    )
    parser.add_argument(
        "-nc", "--no-clean",
        action="store_true",
        default=False,
        help="Skip protein cleaning before conversion. By default, cleaning is enabled "
        "(remove non-standard residues, hydrogens, and alternative conformations).",
    )
    return parser


def main(args: argparse.Namespace | None = None) -> str:
    """CLI entry point."""
    parser = create_parser()
    if args is None:
        args = parser.parse_args()
    
    try:
        pdbqt_path = prepare_receptor(
            input_pdb=args.input,
            output_path=args.output,
            clean=not args.no_clean,
            remove_hydrogens=args.remove_h,
        )
        h_status = "no H" if args.remove_h else "with H"
        clean_status = "cleaned" if not args.no_clean else "not cleaned"
        print(f"Successfully generated PDBQT: {pdbqt_path} ({h_status}, {clean_status})")
        return pdbqt_path
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
