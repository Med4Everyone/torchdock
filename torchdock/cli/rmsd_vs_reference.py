#!/usr/bin/env python3
"""
RMSD calculation — compare predicted docking poses against a reference.

Supports both semi-flexible (``result_remi.pdbqt``) and flexible
(``result.pdbqt``) TorchDock results.  The tool auto-detects multi-model
files and computes RMSD for every pose (or the top-*k* poses).

RMSD is calculated with symmetry-aware matching (``spyrmsd``) on heavy
atoms only, consistent with the TorchDock evaluation pipeline.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import argparse
import os
import re
import shutil
import sys
import tempfile

from torchdock.config.config import Config
from torchdock.data.vina_dataloader import VinaLigandLoader
from torchdock.metrics.rmsd import calculate_rmsd
from torchdock.utils.logging import setup_logger


# ---------------------------------------------------------------------------
# PDBQT model parsing (inlined to avoid external split_pdbqt_models dep)
# ---------------------------------------------------------------------------


def _parse_ligand_models(pdbqt_path: str) -> list[dict]:
    """Parse a multi-model PDBQT result file, extracting ligand sections.

    Only the ligand portion (MODEL header through TORSDOF) is kept; any
    pocket / receptor atoms are discarded.

    Args:
        pdbqt_path: Path to the PDBQT result file.

    Returns:
        List of dicts, each with keys ``score`` (float | None, VINA RESULT
        first column) and ``ligand_text`` (str, full ligand PDBQT block
        including MODEL/ENDMDL).

    Raises:
        ValueError: If no MODEL/ENDMDL blocks are found in the file.
    """
    with open(pdbqt_path, "r") as fh:
        content = fh.read()

    model_pattern = re.compile(
        r"^(MODEL\s+\d+\s*\n)(.*?)^ENDMDL",
        re.MULTILINE | re.DOTALL,
    )
    matches = model_pattern.findall(content)
    if not matches:
        raise ValueError(f"No MODEL/ENDMDL blocks found in: {pdbqt_path}")

    models = []
    for header, block in matches:
        lines = block.splitlines(keepends=True)
        score = None
        ligand_lines: list[str] = [header]
        in_pocket = False

        for line in lines:
            if line.startswith("REMARK VINA RESULT:"):
                parts = line.split()
                score = float(parts[3])

            if "REMARK POCKET ATOMS START" in line:
                in_pocket = True
                continue
            if "REMARK POCKET ATOMS END" in line:
                in_pocket = False
                continue

            # Skip pocket atoms
            if in_pocket:
                continue

            ligand_lines.append(line)

        # Ensure ENDMDL is present
        ligand_lines.append("ENDMDL\n")
        models.append({
            "score": score,
            "ligand_text": "".join(ligand_lines),
        })

    return models


# ---------------------------------------------------------------------------
# RMSD calculation helpers
# ---------------------------------------------------------------------------


def _load_ligand(pdbqt_path: str, score_function: str = "vina") -> VinaLigandLoader:
    """Load a PDBQT ligand file via VinaLigandLoader.

    Args:
        pdbqt_path: Path to a single-model PDBQT file.
        score_function: Scoring function name (``vina`` or ``vinardo``).

    Returns:
        Loaded VinaLigandLoader instance.
    """
    config = Config()
    logger = setup_logger(name="rmsd_calc", level="ERROR")
    config.setattr("logger", logger)
    config.setattr("score_function", score_function)
    config.setattr("ligand_file_path", pdbqt_path)
    return VinaLigandLoader(config)


def _compute_rmsd(
    pred_loader: VinaLigandLoader,
    ref_loader: VinaLigandLoader,
) -> float:
    """Compute symmetry-aware RMSD between two loaded molecules.

    Args:
        pred_loader: Predicted pose loader.
        ref_loader: Reference structure loader.

    Returns:
        RMSD value in Angstrom.
    """
    return calculate_rmsd(
        pred_loader.ligand_coords,
        ref_loader.ligand_coords,
        pred_loader.atomicnums,
        ref_loader.atomicnums,
        pred_loader.adjacency_matrix,
        ref_loader.adjacency_matrix,
        consider_symmetry=True,
        ignore_hydrogen=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_rmsd_vs_reference(
    predicted_path: str,
    reference_path: str,
    top_k: int | None = None,
    score_function: str = "vina",
    verbose: bool = True,
) -> list[tuple[int, float, float | None]]:
    """Calculate RMSD between predicted pose(s) and a reference structure.

    Args:
        predicted_path: Path to predicted PDBQT (single or multi-model).
        reference_path: Path to reference PDBQT (single pose).
        top_k: Only evaluate the top *k* models. ``None`` means all.
        score_function: Scoring function for VinaLigandLoader.
        verbose: Print progress.

    Returns:
        List of ``(rank, rmsd, score)`` tuples.
    """
    # Load reference once
    if verbose:
        print(f"Loading reference: {reference_path}")
    ref_loader = _load_ligand(reference_path, score_function)

    # Parse predicted models
    models = _parse_ligand_models(predicted_path)
    if top_k is not None:
        models = models[:top_k]

    if verbose:
        print(f"Found {len(models)} model(s) in: {predicted_path}")

    # Create temp directory for individual ligand files
    temp_dir = tempfile.mkdtemp(prefix="torchdock_rmsd_")
    results: list[tuple[int, float, float | None]] = []

    try:
        for rank, model in enumerate(models, start=1):
            # Write individual ligand PDBQT
            ligand_file = os.path.join(temp_dir, f"ligand_{rank}.pdbqt")
            with open(ligand_file, "w") as fh:
                fh.write(model["ligand_text"])

            pred_loader = _load_ligand(ligand_file, score_function)
            rmsd_val = _compute_rmsd(pred_loader, ref_loader)
            results.append((rank, rmsd_val, model["score"]))

            if verbose:
                score_str = (
                    f"{model['score']:.3f}" if model["score"] is not None else "N/A"
                )
                print(f"  Rank {rank:>2d}: RMSD = {rmsd_val:.3f} Å  (score = {score_str})")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Calculate RMSD between predicted docking poses and a reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Multi-model result vs reference
  torchdock calc-rmsd -p result_remi.pdbqt -r reference.pdbqt

  # Only top 3 poses
  torchdock calc-rmsd -p result.pdbqt -r reference.pdbqt -t 3

  # Quiet mode (prints only RMSD values)
  torchdock calc-rmsd -p result.pdbqt -r reference.pdbqt -q
""",
    )
    parser.add_argument(
        "-p", "--predicted",
        type=str,
        required=True,
        help="Predicted PDBQT result file (single or multi-model).",
    )
    parser.add_argument(
        "-r", "--reference",
        type=str,
        required=True,
        help="Reference structure PDBQT file (single pose).",
    )
    parser.add_argument(
        "-t", "--top-k",
        type=int,
        default=None,
        help="Only evaluate top-k models by rank. Default: all.",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Quiet mode — only print RMSD values, one per line.",
    )
    return parser


def main(args: argparse.Namespace | None = None) -> list[tuple[int, float, float | None]]:
    """CLI entry point."""
    parser = create_parser()
    if args is None:
        args = parser.parse_args()

    verbose = not args.quiet

    try:
        # Validate inputs
        for path, label in [
            (args.predicted, "Predicted"),
            (args.reference, "Reference"),
        ]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{label} file not found: {path}")

        if verbose:
            print("=" * 60)
            print("RMSD Calculation (symmetry-aware, heavy atoms only)")
            print("=" * 60)

        results = calculate_rmsd_vs_reference(
            predicted_path=args.predicted,
            reference_path=args.reference,
            top_k=args.top_k,
            verbose=verbose,
        )

        if verbose:
            print("=" * 60)
            print("Summary:")
            for rank, rmsd_val, score in results:
                score_str = f"{score:.3f}" if score is not None else "N/A"
                print(f"  Rank {rank:>2d}: RMSD = {rmsd_val:.3f} Å  (score = {score_str})")
            print("=" * 60)
        else:
            # Quiet mode: just RMSD values
            for _, rmsd_val, _ in results:
                print(f"{rmsd_val:.3f}")

        return results

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
