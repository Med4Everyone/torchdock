#!/usr/bin/env python3
"""
Docking box definition — generate box JSON and optional PDB visualization.

Supports two modes:
    1. Ligand-based: automatically compute box center from heavy-atom
       coordinates of a ligand file (MOL2, SDF, PDB, or PDBQT).
    2. Manual: specify the box center explicitly via ``--center``.

The output is a JSON file containing ``center`` and ``size`` arrays.
With ``-v``, a PDB file visualising the box wireframe is also written.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from rdkit import Chem
    from rdkit import RDLogger

    _rdkit_logger = RDLogger.logger()
except ImportError:
    Chem = None
    _rdkit_logger = None

try:
    from openbabel import pybel
except ImportError:
    pybel = None

# Supported ligand file extensions mapped to their parsers.
_RDKIT_FORMATS = {".mol2": "mol2", ".sdf": "sdf", ".pdb": "pdb"}


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------


def _check_deps(ligand_path: str | None) -> None:
    """Check that required dependencies are installed.

    Args:
        ligand_path: If provided, the file extension determines which
            parser library is required (RDKit or OpenBabel).

    Raises:
        ImportError: If the required library is not installed.
    """
    if ligand_path is None:
        return
    ext = Path(ligand_path).suffix.lower()
    if ext in _RDKIT_FORMATS and Chem is None:
        raise ImportError(
            "Reading MOL2/SDF files requires rdkit. "
            "Install with: pip install rdkit"
        )
    if ext == ".pdbqt" and pybel is None:
        raise ImportError(
            "Reading PDBQT files requires openbabel. "
            "Install with: pip install openbabel-wheel"
        )


# ---------------------------------------------------------------------------
# Ligand coordinate reading
# ---------------------------------------------------------------------------


def _read_heavy_atom_coords(ligand_path: str) -> np.ndarray:
    """Read heavy-atom (non-hydrogen) 3D coordinates from a ligand file.

    Supported formats: MOL2, SDF (via RDKit, sanitize=False), and
    PDBQT (via OpenBabel).

    Args:
        ligand_path: Path to the ligand file.

    Returns:
        Heavy-atom coordinates, shape ``(N, 3)``.

    Raises:
        ValueError: If the file format is unsupported or parsing fails.
    """
    ext = Path(ligand_path).suffix.lower()

    if ext in _RDKIT_FORMATS:
        return _read_heavy_atom_coords_rdkit(ligand_path, ext)
    if ext == ".pdbqt":
        return _read_heavy_atom_coords_openbabel(ligand_path)

    raise ValueError(
        f"Unsupported ligand format: '{ext}'. "
        f"Supported: {', '.join(sorted(_RDKIT_FORMATS)) + ', .pdbqt'}"
    )


def _read_heavy_atom_coords_rdkit(
    ligand_path: str, ext: str
) -> np.ndarray:
    """Read heavy-atom coordinates via RDKit with sanitization disabled.

    Args:
        ligand_path: Path to the MOL2 or SDF file.
        ext: Lower-case file extension (``.mol2`` or ``.sdf``).

    Returns:
        Heavy-atom coordinates, shape ``(N, 3)``.

    Raises:
        ValueError: If RDKit fails to parse the file or no conformer is found.
    """
    _rdkit_logger.setLevel(RDLogger.WARNING)
    fmt = _RDKIT_FORMATS[ext]

    if fmt == "mol2":
        mol = Chem.MolFromMol2File(ligand_path, sanitize=False)
    elif fmt == "pdb":
        mol = Chem.MolFromPDBFile(ligand_path, sanitize=False)
    else:
        supplier = Chem.SDMolSupplier(ligand_path, sanitize=False)
        mol = next(iter(supplier), None)

    if mol is None:
        raise ValueError(f"RDKit failed to parse file: {ligand_path}")
    if mol.GetNumConformers() == 0:
        raise ValueError(f"No 3D conformer found in: {ligand_path}")

    conf = mol.GetConformer()
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() > 1:  # Skip hydrogen atoms.
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])

    if not coords:
        raise ValueError(f"No heavy atoms found in: {ligand_path}")
    return np.array(coords, dtype=np.float64)


def _read_heavy_atom_coords_openbabel(ligand_path: str) -> np.ndarray:
    """Read heavy-atom coordinates via OpenBabel.

    Args:
        ligand_path: Path to the PDBQT file.

    Returns:
        Heavy-atom coordinates, shape ``(N, 3)``.

    Raises:
        ValueError: If parsing fails or no heavy atoms are found.
    """
    mol = next(pybel.readfile("pdbqt", ligand_path), None)
    if mol is None:
        raise ValueError(f"OpenBabel failed to parse file: {ligand_path}")

    coords = []
    for atom in mol.atoms:
        if atom.atomicnum > 1:  # Skip hydrogen atoms.
            coords.append(list(atom.coords))

    if not coords:
        raise ValueError(f"No heavy atoms found in: {ligand_path}")
    return np.array(coords, dtype=np.float64)


# ---------------------------------------------------------------------------
# Box computation
# ---------------------------------------------------------------------------


def _compute_center(coords: np.ndarray) -> np.ndarray:
    """Compute the geometric center (centroid) of coordinates.

    Args:
        coords: Atomic coordinates, shape ``(N, 3)``.

    Returns:
        Centroid, shape ``(3,)``.
    """
    return np.mean(coords, axis=0)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _save_box_json(
    center: np.ndarray, size: np.ndarray, output_path: str
) -> None:
    """Save docking box parameters to a JSON file.

    Args:
        center: Box center coordinates, shape ``(3,)``.
        size: Box dimensions, shape ``(3,)``.
        output_path: Path to the output JSON file.
    """
    box_info = {
        "center": center.tolist(),
        "size": size.tolist(),
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(box_info, f, indent=4)


def _generate_box_pdb(
    center: np.ndarray, size: np.ndarray, output_path: str
) -> None:
    """Generate a PDB file visualising the docking box wireframe.

    Creates 8 vertex atoms (Ne) connected by CONECT records to form a
    rectangular wireframe, plus a center atom (Xe) for reference.

    Args:
        center: Box center coordinates, shape ``(3,)``.
        size: Box dimensions, shape ``(3,)``.
        output_path: Path to the output PDB file.
    """
    cx, cy, cz = center
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2

    vertices = [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]

    edges = [
        (1, 2), (1, 4), (1, 5),
        (2, 3), (2, 6),
        (3, 4), (3, 7),
        (4, 8),
        (5, 6), (5, 8),
        (6, 7),
        (7, 8),
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        # Vertex atoms (Ne).
        for i, (x, y, z) in enumerate(vertices, 1):
            f.write(
                f"ATOM  {i:5d}  Ne  BOX X{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00          Ne\n"
            )
        # Center atom (Xe).
        f.write(
            f"ATOM  {9:5d}  Xe  BOX X   9    "
            f"{cx:8.3f}{cy:8.3f}{cz:8.3f}  1.00 10.00          Xe\n"
        )
        # Connectivity.
        for a1, a2 in edges:
            f.write(f"CONECT{a1:5d}{a2:5d}\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def define_box(
    ligand_path: str | None = None,
    center: list[float] | None = None,
    size: list[float] | None = None,
    output_path: str = "box.json",
    visualize: bool = False,
) -> str:
    """Define a docking box and write output files.

    Exactly one of ``ligand_path`` or ``center`` must be provided.
    When ``ligand_path`` is given, the box center is computed as the
    centroid of the ligand's heavy atoms.

    Args:
        ligand_path: Path to a ligand file (MOL2, SDF, or PDBQT).
        center: Manual box center ``[x, y, z]``.
        size: Box dimensions ``[sx, sy, sz]``. Defaults to ``[20, 20, 20]``.
        output_path: Output JSON file path.
        visualize: If True, also write a PDB wireframe file using the
            same stem as *output_path* with a ``.pdb`` extension.

    Returns:
        Path to the generated JSON file.

    Raises:
        ValueError: If neither or both of ligand_path / center are given.
    """
    if (ligand_path is None) == (center is None):
        raise ValueError(
            "Exactly one of --ligand or --center must be specified."
        )

    if size is None:
        size = [20.0, 20.0, 20.0]

    # Determine box center.
    if ligand_path is not None:
        _check_deps(ligand_path)
        coords = _read_heavy_atom_coords(ligand_path)
        box_center = _compute_center(coords)
        print(
            f"Ligand center: [{box_center[0]:.3f}, {box_center[1]:.3f}, "
            f"{box_center[2]:.3f}] ({len(coords)} heavy atoms)",
            file=sys.stderr,
        )
    else:
        box_center = np.array(center, dtype=np.float64)

    box_size = np.array(size, dtype=np.float64)

    print(
        f"Box center: [{box_center[0]:.3f}, {box_center[1]:.3f}, {box_center[2]:.3f}]",
        file=sys.stderr,
    )
    print(
        f"Box size:   [{box_size[0]:.3f}, {box_size[1]:.3f}, {box_size[2]:.3f}]",
        file=sys.stderr,
    )

    # Write JSON.
    _save_box_json(box_center, box_size, output_path)

    # Optionally write visualization PDB.
    if visualize:
        pdb_path = str(Path(output_path).with_suffix(".pdb"))
        _generate_box_pdb(box_center, box_size, pdb_path)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Define a docking box from a ligand file or manual coordinates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Auto-center from a ligand file
  torchdock define-box -l ligand.pdbqt -o box.json

  # Auto-center with visualization PDB
  torchdock define-box -l ligand.sdf -o box.json -v

  # Manual center with custom size
  torchdock define-box -c 10.0 20.0 30.0 -s 25 25 25 -o box.json
""",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-l", "--ligand",
        type=str,
        help="Ligand file (MOL2, SDF, PDB, or PDBQT) to auto-compute center "
        "from heavy atoms.",
    )
    input_group.add_argument(
        "-c", "--center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Manual box center coordinates.",
    )

    parser.add_argument(
        "-s", "--size",
        type=float,
        nargs=3,
        default=[20.0, 20.0, 20.0],
        metavar=("SX", "SY", "SZ"),
        help="Box dimensions (default: 20 20 20).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        default=False,
        help="Also generate a PDB visualization of the box "
        "(same path as -o with .pdb extension).",
    )
    return parser


def main(args: argparse.Namespace | None = None) -> str:
    """CLI entry point."""
    parser = create_parser()
    if args is None:
        args = parser.parse_args()

    try:
        result = define_box(
            ligand_path=args.ligand,
            center=args.center,
            size=args.size,
            output_path=args.output,
            visualize=args.visualize,
        )
        print(f"Box JSON generated: {result}")
        return result
    except (ImportError, ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
