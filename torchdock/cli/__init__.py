"""
TorchDock CLI: command-line tools for molecular docking.

Usage:
    torchdock dock              [options]
    torchdock prepare_ligand    [options]
    torchdock prepare_receptor  [options]
    torchdock define_box        [options]
    torchdock convert_result    [options]
    torchdock rmsd              [options]
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import sys

# Subcommand -> (module_path, description)
_SUBCOMMANDS = {
    "dock": (
        "torchdock.pipeline.docking_runner",
        "Run molecular docking.",
    ),
    "prepare_ligand": (
        "torchdock.cli.prepare_ligand",
        "Convert SMILES or file input to PDBQT ligand.",
    ),
    "prepare_receptor": (
        "torchdock.cli.prepare_receptor",
        "Prepare receptor PDBQT from a PDB file.",
    ),
    "define_box": (
        "torchdock.cli.define_box",
        "Define a docking box from a ligand or manual coordinates.",
    ),
    "convert_result": (
        "torchdock.cli.convert_result",
        "Convert TorchDock PDBQT results to SDF and PDB.",
    ),
    "rmsd": (
        "torchdock.cli.rmsd_vs_reference",
        "Calculate RMSD between docking poses and a reference.",
    ),
}


def main() -> None:
    """Entry point for the ``torchdock`` command."""
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _print_usage()
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in _SUBCOMMANDS:
        print(f"Unknown command: {cmd}\n")
        _print_usage()
        sys.exit(1)

    # Lazy import to keep startup fast
    import importlib

    module = importlib.import_module(_SUBCOMMANDS[cmd][0])

    # Strip the subcommand name so each module's parser sees only its own args
    sys.argv = [f"torchdock {cmd}"] + sys.argv[2:]
    module.main()


def _print_usage() -> None:
    """Print top-level help."""
    from torchdock import __version__

    print(f"TorchDock v{__version__} — Differentiable molecular docking framework\n")
    print("Usage: torchdock <command> [options]\n")
    print("Commands:")
    for name, (_, desc) in _SUBCOMMANDS.items():
        print(f"  {name:<20s} {desc}")
    print(f"\nRun 'torchdock <command> --help' for command-specific options.")
