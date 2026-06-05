"""
RMSD calculation utilities for molecular conformations.

Provides functions to compute root-mean-square deviation between molecular
structures, with support for symmetry-aware RMSD and hydrogen atom filtering.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import numpy as np
import torch
from spyrmsd import rmsd


def calculate_rmsd(
    coords1: torch.Tensor | np.ndarray,
    coords2: torch.Tensor | np.ndarray,
    atomicnums1: np.ndarray,
    atomicnums2: np.ndarray,
    adjacency_matrix1: np.ndarray,
    adjacency_matrix2: np.ndarray,
    consider_symmetry: bool = True,
    ignore_hydrogen: bool = True,
) -> float:
    """Calculate RMSD between two molecular conformations.

    Args:
        coords1: First conformation coordinates, shape (N1, 3).
        coords2: Second conformation coordinates, shape (N2, 3).
        atomicnums1: Atomic numbers of the first molecule.
        atomicnums2: Atomic numbers of the second molecule.
        adjacency_matrix1: Adjacency matrix of the first molecule.
        adjacency_matrix2: Adjacency matrix of the second molecule.
        consider_symmetry: If True, calculate symmetric RMSD (lower bound)
            considering molecular symmetry. Defaults to True.
        ignore_hydrogen: If True, ignore hydrogen atoms (atomic number = 1)
            in RMSD calculation. Defaults to True.

    Returns:
        RMSD value between the two conformations.

    Raises:
        ValueError: If heavy atom type counts mismatch or atom counts differ
            after filtering.
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(coords1):
        coords1 = coords1.detach().cpu().numpy()
    if torch.is_tensor(coords2):
        coords2 = coords2.detach().cpu().numpy()
    
    # Filter out hydrogen atoms if requested
    if ignore_hydrogen:
        mask1 = atomicnums1 != 1
        mask2 = atomicnums2 != 1
        idx1 = np.where(mask1)[0]
        idx2 = np.where(mask2)[0]
        
        # Filter coordinates
        coords1 = coords1[idx1]
        coords2 = coords2[idx2]
        
        # Filter atomic numbers
        atomicnums1 = atomicnums1[idx1]
        atomicnums2 = atomicnums2[idx2]
        
        # Filter adjacency matrices (keep only heavy atom connections)
        adjacency_matrix1 = adjacency_matrix1[np.ix_(idx1, idx1)]
        adjacency_matrix2 = adjacency_matrix2[np.ix_(idx2, idx2)]

    # Check heavy atom type counts consistency (exclude hydrogens)
    def _counts(arr):
        vals, counts = np.unique(arr[arr != 1], return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    heavy_counts_1 = _counts(atomicnums1)
    heavy_counts_2 = _counts(atomicnums2)
    if heavy_counts_1 != heavy_counts_2:
        raise ValueError(
            f"Heavy atom type counts mismatch: first={heavy_counts_1}, second={heavy_counts_2}"
        )

    # Optional sanity check on total heavy atom count
    if coords1.shape[0] != coords2.shape[0]:
        raise ValueError(
            f"Number of atoms after filtering must match: first={coords1.shape[0]}, second={coords2.shape[0]}"
        )

    if consider_symmetry:
        # Calculate symmetric RMSD (lower bound) using molecule-specific topology
        return rmsd.symmrmsd(
            coords1,
            coords2,
            atomicnums1,
            atomicnums2,
            adjacency_matrix1,
            adjacency_matrix2
        )
    else:
        # Calculate standard RMSD (upper bound) without considering symmetry
        return rmsd.rmsd(coords1, coords2, atomicnums1, atomicnums2)
