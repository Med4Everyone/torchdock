#!/usr/bin/env python3
"""
Result conversion — convert TorchDock PDBQT results to SDF (ligand) and PDB (pocket).

Supports two result types:
    1. Semi-flexible docking (result_remi.pdbqt): ligand only.
    2. Flexible docking (result.pdbqt): ligand + pocket atoms.

The tool auto-detects the result type by checking for ``REMARK POCKET ATOMS START``.
Each model is converted and saved with a rank-based filename.

Output naming convention:
    - Ligand: ``rank_{rank}_{score}_ligand.sdf``
    - Pocket (flex only): ``rank_{rank}_{score}_pocket.pdb``
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

try:
    from openbabel import openbabel as ob
    from openbabel import pybel
except ImportError:
    ob = None
    pybel = None


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------


def _check_deps() -> None:
    """Check that OpenBabel is installed."""
    if ob is None or pybel is None:
        raise ImportError(
            "convert-result requires openbabel. "
            "Install it with: conda install -c conda-forge openbabel"
        )


# ---------------------------------------------------------------------------
# PDBQT model parsing
# ---------------------------------------------------------------------------


def _parse_models(pdbqt_path: str) -> list[dict]:
    """Parse a multi-model PDBQT file into individual model dicts.

    Each dict contains:
        - ``score`` (float): VINA RESULT score (first column).
        - ``ligand_lines`` (list[str]): ATOM/HETATM lines for the ligand.
        - ``pocket_lines`` (list[str] | None): ATOM/HETATM lines for the
          pocket, or *None* when the file has no pocket section.

    Args:
        pdbqt_path: Path to the input PDBQT result file.

    Returns:
        List of model dicts, one per MODEL block.
    """
    with open(pdbqt_path, "r") as fh:
        content = fh.read()

    # Split by MODEL blocks
    model_pattern = re.compile(
        r"^MODEL\s+\d+\s*\n(.*?)^ENDMDL",
        re.MULTILINE | re.DOTALL,
    )
    matches = model_pattern.findall(content)
    if not matches:
        raise ValueError(f"No MODEL/ENDMDL blocks found in: {pdbqt_path}")

    models = []
    for block in matches:
        lines = block.splitlines(keepends=True)
        score = None
        ligand_lines: list[str] = []
        pocket_lines: list[str] = []
        in_pocket = False

        for line in lines:
            # Extract VINA score
            if line.startswith("REMARK VINA RESULT:"):
                parts = line.split()
                # Format: REMARK VINA RESULT:  score  rmsd_lb  rmsd_ub
                score = float(parts[3])
                continue

            # Pocket boundary markers
            if "REMARK POCKET ATOMS START" in line:
                in_pocket = True
                continue
            if "REMARK POCKET ATOMS END" in line:
                in_pocket = False
                continue

            # Skip non-atom lines (REMARK, ROOT, ENDROOT, BRANCH, etc.)
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            if in_pocket:
                pocket_lines.append(line)
            else:
                ligand_lines.append(line)

        models.append({
            "score": score,
            "ligand_lines": ligand_lines,
            "pocket_lines": pocket_lines if pocket_lines else None,
        })

    return models


def _is_flex_result(models: list[dict]) -> bool:
    """Check whether models contain pocket atoms (flexible docking)."""
    return any(m["pocket_lines"] is not None for m in models)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _atom_lines_to_pdbqt(lines: list[str], is_ligand: bool) -> str:
    """Build a minimal PDBQT string from raw ATOM/HETATM lines.

    For ligands the torsion-tree keywords (ROOT, TORSDOF, …) are not
    needed — OpenBabel can parse flat ATOM records.

    Args:
        lines: Raw ATOM/HETATM lines from the PDBQT file.
        is_ligand: If *True*, wrap with ``ROOT`` / ``ENDROOT`` / ``TORSDOF``.

    Returns:
        A PDBQT-formatted string.
    """
    parts: list[str] = []
    if is_ligand:
        parts.append("ROOT\n")
    for line in lines:
        parts.append(line if line.endswith("\n") else line + "\n")
    if is_ligand:
        parts.append("ENDROOT\n")
        parts.append("TORSDOF 0\n")
    return "".join(parts)


def _convert_ligand(lines: list[str], output_sdf: str) -> None:
    """Convert ligand ATOM lines to SDF via OpenBabel.

    The conversion goes: PDBQT → MOL2 → SDF to ensure correct bond
    perception (consistent with the prepare-ligand pipeline).

    Args:
        lines: Ligand ATOM/HETATM lines.
        output_sdf: Destination SDF file path.
    """
    pdbqt_str = _atom_lines_to_pdbqt(lines, is_ligand=True)

    with tempfile.NamedTemporaryFile(
        suffix=".pdbqt", mode="w", delete=False
    ) as tmp_pdbqt:
        tmp_pdbqt.write(pdbqt_str)
        tmp_pdbqt_path = tmp_pdbqt.name

    try:
        mol = next(pybel.readfile("pdbqt", tmp_pdbqt_path), None)
        if mol is None:
            raise RuntimeError("OpenBabel failed to parse ligand PDBQT block.")

        # Clear the temp-file title so the SDF header is clean
        mol.title = ""

        # PDBQT → MOL2 intermediate for reliable bond perception
        with tempfile.NamedTemporaryFile(
            suffix=".mol2", mode="w", delete=False
        ) as tmp_mol2:
            tmp_mol2_path = tmp_mol2.name

        mol.write("mol2", tmp_mol2_path, overwrite=True)
        mol2 = next(pybel.readfile("mol2", tmp_mol2_path), None)
        if mol2 is None:
            # Fallback: direct write if MOL2 round-trip fails
            mol.write("sdf", output_sdf, overwrite=True)
        else:
            mol2.title = ""
            mol2.write("sdf", output_sdf, overwrite=True)
        os.unlink(tmp_mol2_path)
    finally:
        os.unlink(tmp_pdbqt_path)


def _convert_pocket(lines: list[str], output_pdb: str) -> None:
    """Convert pocket ATOM lines to PDB via OpenBabel.

    Args:
        lines: Pocket ATOM/HETATM lines from the PDBQT file.
        output_pdb: Destination PDB file path.
    """
    pdbqt_str = _atom_lines_to_pdbqt(lines, is_ligand=False)

    with tempfile.NamedTemporaryFile(
        suffix=".pdbqt", mode="w", delete=False
    ) as tmp_pdbqt:
        tmp_pdbqt.write(pdbqt_str)
        tmp_pdbqt_path = tmp_pdbqt.name

    try:
        mol = next(pybel.readfile("pdbqt", tmp_pdbqt_path), None)
        if mol is None:
            raise RuntimeError("OpenBabel failed to parse pocket PDBQT block.")
        mol.title = ""
        mol.write("pdb", output_pdb, overwrite=True)
    finally:
        os.unlink(tmp_pdbqt_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_result(
    input_path: str,
    output_dir: str,
    top_k: int | None = None,
) -> list[str]:
    """Convert TorchDock PDBQT result to SDF/PDB files.

    Args:
        input_path: Path to the PDBQT result file (flex or semi-flex).
        output_dir: Directory to write converted files.
        top_k: Only convert the top *k* models (by rank). ``None`` means all.

    Returns:
        List of generated file paths.
    """
    _check_deps()

    input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    models = _parse_models(input_path)
    has_pocket = _is_flex_result(models)

    if top_k is not None:
        models = models[:top_k]

    generated: list[str] = []

    for rank, model in enumerate(models, start=1):
        score = model["score"]
        if score is None:
            print(
                f"Warning: MODEL {rank} has no VINA score, skipping.",
                file=sys.stderr,
            )
            continue

        # Format score: keep 2 decimal places, use underscore for negative sign
        score_str = f"{score:.2f}"

        # --- Ligand ---
        ligand_name = f"rank_{rank}_{score_str}_ligand.sdf"
        ligand_path = os.path.join(output_dir, ligand_name)
        try:
            _convert_ligand(model["ligand_lines"], ligand_path)
            generated.append(ligand_path)
            print(f"  [{rank:>2d}] Ligand → {ligand_name}")
        except Exception as e:
            print(
                f"  [{rank:>2d}] Warning: ligand conversion failed — {e}",
                file=sys.stderr,
            )

        # --- Pocket (flex results only) ---
        if has_pocket and model["pocket_lines"] is not None:
            pocket_name = f"rank_{rank}_{score_str}_pocket.pdb"
            pocket_path = os.path.join(output_dir, pocket_name)
            try:
                _convert_pocket(model["pocket_lines"], pocket_path)
                generated.append(pocket_path)
                print(f"  [{rank:>2d}] Pocket → {pocket_name}")
            except Exception as e:
                print(
                    f"  [{rank:>2d}] Warning: pocket conversion failed — {e}",
                    file=sys.stderr,
                )

    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert TorchDock PDBQT results to SDF (ligand) and PDB (pocket).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Convert all results (semi-flexible)
  torchdock convert-result -i result_remi.pdbqt -o ./output

  # Convert top 5 results (flexible docking)
  torchdock convert-result -i result.pdbqt -o ./output -t 5

  # Convert all results (flexible docking, outputs both ligand SDF and pocket PDB)
  torchdock convert-result -i result.pdbqt -o ./output
""",
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input PDBQT result file (result.pdbqt or result_remi.pdbqt).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output directory for converted files.",
    )
    parser.add_argument(
        "-t", "--top-k",
        type=int,
        default=None,
        help="Only convert the top-k models by rank. Default: all.",
    )
    return parser


def main(args: argparse.Namespace | None = None) -> list[str]:
    """CLI entry point."""
    parser = create_parser()
    if args is None:
        args = parser.parse_args()

    try:
        input_path = args.input
        result_type = "flex" if "remi" not in os.path.basename(input_path) else "semi-flex"
        print(f"Converting TorchDock results ({result_type}): {input_path}")

        generated = convert_result(
            input_path=input_path,
            output_dir=args.output,
            top_k=args.top_k,
        )
        print(f"\nDone — {len(generated)} file(s) written to: {args.output}")
        return generated
    except (ImportError, ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
