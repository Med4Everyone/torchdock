"""
Base data loader classes for molecular docking.

This module defines abstract base classes for ligand and protein loaders,
providing shared utilities for coordinate validation, torsion extraction,
connectivity analysis, and flexible residue handling.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import copy
from abc import ABC, abstractmethod

import numpy as np

from torchdock.constants.atom14_constants import (
    CHI_ANGLES_ATOM14_INDICES,
    NON_FLEXIBLE_RESIDUES,
    RESTYPE_NAME_TO_ATOM14_NAMES,
)


class BaseLigandLoader(ABC):
    """Abstract base class for ligand data loaders.

    Provides shared utilities for coordinate validation, torsion extraction,
    connectivity analysis, and PDBQT file I/O. Subclasses must implement
    the load_molecule() method.
    """

    def __init__(self, config: object) -> None:
        self.config = config
        self.logger = config.logger
        self.dtype = np.dtype(config.dtype)  # No actual use in data loading.

        self.load_molecule()

    @abstractmethod
    def load_molecule(self):
        raise NotImplementedError

    def _validate_ligand_coordinates(
        self, ligand_coords: np.ndarray, num_atoms: int, tol: float = 1e-6
    ) -> None:
        """Validate ligand coordinates for NaN/Inf values and duplicate/overlapping atoms.

        Args:
            ligand_coords: Ligand atom coordinates, shape (N, 3).
            num_atoms: Number of atoms in the ligand.
            tol: Tolerance for detecting duplicate coordinates. Defaults to 1e-6.

        Raises:
            ValueError: If coordinates contain NaN/Inf or duplicate/overlapping atoms.
        """
        # Check for NaN/Inf values
        if not np.all(np.isfinite(ligand_coords)):
            self.logger.error("Ligand coordinates contain NaN/Inf values")
            raise ValueError("Ligand coordinates contain NaN/Inf values")

        # Check for duplicate coordinates (within tolerance)
        if num_atoms > 1:
            diffs = ligand_coords[:, None, :] - ligand_coords[None, :, :]
            d2 = np.sum(diffs * diffs, axis=-1)
            tri_mask = np.triu(np.ones((num_atoms, num_atoms), dtype=bool), 1)
            pair_d2 = d2[tri_mask]
            if np.any(pair_d2 < tol * tol):
                # Find a few offending pairs for debugging
                dup_mask = (d2 < tol * tol) & tri_mask
                dup_i, dup_j = np.where(dup_mask)
                msg = f"Found duplicate/overlapping atom coordinates within tolerance {tol}. Examples: "
                examples = []
                for k in range(min(5, len(dup_i))):
                    i_idx, j_idx = int(dup_i[k]), int(dup_j[k])
                    examples.append(f"({i_idx},{j_idx})")
                msg += ", ".join(examples)
                self.logger.error(msg)
                raise ValueError("Ligand has duplicate/overlapping atom coordinates")

    def _check_ligand_connectivity(self, bond_matrix: np.ndarray) -> None:
        """Check if the ligand is a single connected component using DFS.

        Args:
            bond_matrix: Connectivity matrix, shape (N, N). A positive value at
                [i][j] indicates atoms i and j are bonded.

        Raises:
            ValueError: If the ligand contains disconnected fragments.
        """
        num_atoms = bond_matrix.shape[0]
        
        # Handle single-atom case (always connected)
        if num_atoms == 1:
            return
        
        # Build adjacency list from bond matrix for efficient traversal
        adjacency = [[] for _ in range(num_atoms)]
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if bond_matrix[i][j] > 0:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
        
        # Perform DFS from atom 0 to find all reachable atoms
        visited = np.zeros(num_atoms, dtype=bool)
        stack = [0]
        visited[0] = True
        visited_count = 1
        
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    visited_count += 1
                    stack.append(neighbor)
        
        # Check if all atoms are reachable (single connected component)
        if visited_count != num_atoms:
            unvisited_atoms = np.where(~visited)[0]
            msg = (
                f"Ligand contains disconnected fragments! "
                f"Found {visited_count} atoms in main component, "
                f"{num_atoms - visited_count} atoms isolated. "
                f"First isolated atom index: {unvisited_atoms[0]}"
            )
            self.logger.error(msg)
            raise ValueError("Ligand must be a single connected molecule without isolated fragments")

    def build_rdkit_to_pdbqt_mapping(
        self, rdkit_coords: np.ndarray, pdbqt_coords: np.ndarray, tol: float = 1e-4
    ) -> np.ndarray:
        """Build mapping from RDKit atom indices to PDBQT atom indices via coordinate matching.

        Uses a fast path if atoms are already in the same order; otherwise builds an
        explicit mapping using nearest-neighbor matching.

        Args:
            rdkit_coords: Coordinates from RDKit molecule, shape (N, 3).
            pdbqt_coords: Coordinates from PDBQT file (canonical order), shape (N, 3).
            tol: Tolerance for coordinate matching in Angstrom. Defaults to 1e-4.

        Returns:
            Mapping array of shape (N,) where mapping[rdkit_idx] = pdbqt_idx.
            Returns np.arange(N) when the fast path succeeds.

        Raises:
            ValueError: If atom counts differ or mapping cannot be established.
        """
        n_rdkit = rdkit_coords.shape[0]
        n_pdbqt = pdbqt_coords.shape[0]
        
        # Check atom count consistency
        if n_rdkit != n_pdbqt:
            msg = f"Atom count mismatch: RDKit has {n_rdkit} atoms, PDBQT has {n_pdbqt} atoms"
            self.logger.error(msg)
            raise ValueError(msg)
        
        # Fast path: check if atoms are already in the same order
        coord_diffs = rdkit_coords - pdbqt_coords
        max_diff = np.max(np.linalg.norm(coord_diffs, axis=1))
        
        if max_diff < tol:
            # All atoms match in order - use identity mapping
            return np.arange(n_rdkit, dtype=int)
        
        # Slow path: build explicit mapping using nearest neighbor matching
        self.logger.warning(
            f"RDKit and PDBQT atom order differs (max_diff={max_diff:.2e} >= tol={tol}). "
            "Building explicit coordinate-based mapping..."
        )
        
        # Compute pairwise distance matrix: dist[i, j] = distance between rdkit_i and pdbqt_j
        diffs = rdkit_coords[:, None, :] - pdbqt_coords[None, :, :]  # shape (N_rdkit, N_pdbqt, 3)
        dist_matrix = np.linalg.norm(diffs, axis=2)  # shape (N_rdkit, N_pdbqt)
        
        # For each RDKit atom, find closest PDBQT atom
        mapping = np.argmin(dist_matrix, axis=1)  # shape (N_rdkit,)
        min_dists = np.min(dist_matrix, axis=1)
        
        # Validate mapping quality
        # 1. Check that all matches are within tolerance
        if np.any(min_dists > tol):
            bad_indices = np.where(min_dists > tol)[0]
            msg = (
                f"Failed to match {len(bad_indices)} RDKit atoms within tolerance {tol}. "
                f"Examples: RDKit atom {bad_indices[0]} has min_dist={min_dists[bad_indices[0]]:.2e}"
            )
            self.logger.error(msg)
            raise ValueError(msg)
        
        # 2. Check that mapping is one-to-one (no duplicate assignments)
        unique_targets = np.unique(mapping)
        if len(unique_targets) != n_rdkit:
            msg = (
                f"Mapping is not one-to-one: {n_rdkit} RDKit atoms mapped to "
                f"{len(unique_targets)} unique PDBQT atoms. Possible coordinate ambiguity."
            )
            self.logger.error(msg)
            raise ValueError(msg)
        
        # Log statistics
        avg_diff = np.mean(min_dists)
        max_diff_actual = np.max(min_dists)
        self.logger.info(
            f"Built RDKit->PDBQT mapping: avg_diff={avg_diff:.2e}, max_diff={max_diff_actual:.2e}"
        )
        
        return mapping

    def search_connected_atoms(
        self, bond_matrix: np.ndarray, atom_id: int, visited: np.ndarray
    ) -> list[int]:
        """Recursively search for all atoms connected to atom_id that have not been visited.

        Args:
            bond_matrix: Connection matrix where bond_matrix[i][j] > 0 means atoms
                i and j are connected.
            atom_id: Starting atom index.
            visited: Boolean array marking visited atoms.

        Returns:
            List of connected atom indices that are not visited (no duplicates).
        """
        visited[atom_id] = True
        neighbors = np.where(bond_matrix[atom_id] > 0)[0]
        connected_atoms = []
        
        for neighbor in neighbors:
            if not visited[neighbor]:
                # Add the neighbor itself
                connected_atoms.append(neighbor)
                # Recursively search its connected atoms
                sub_connected = self.search_connected_atoms(bond_matrix, neighbor, visited)
                connected_atoms.extend(sub_connected)
        
        return connected_atoms

    def _get_torsions_and_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract torsions and torsion masks using DFS over the molecular graph.

        Only branches containing at least one non-H atom are treated as valid torsions.

        Returns:
            A tuple of:
                - torsions: Torsion atom pairs, shape (N, 2).
                - torsion_masks: Boolean masks for torsion atoms, shape (N, num_atoms).
        """
        # Initialize torsions and masks lists
        torsions = []
        torsion_masks = []
        
        # branch_atoms keys are (parent, child) tuples representing the rotatable bond direction.
        branch_keys = set(self.branch_atoms.keys())

        # Use DFS starting from the first root atom.
        # Note: using self.center_atom directly has a bug — if center_atom is in a branch
        # (e.g., atom 7), DFS cannot traverse back to detect parent->child torsions like (5, 7).
        # In PDBQT files, the first atom typically belongs to the central rigid structure.
        stack = [self.root_atoms[0]]
        visited = np.zeros(self.atoms_num, dtype=bool)
        
        while stack:
            visited[stack[-1]] = True
            cur_atom = stack.pop()
            neighbors = np.where(self.bond_matrix[cur_atom] > 0)[0]
            
            for neighbor in neighbors:
                if not visited[neighbor]:
                    stack.append(neighbor)
                    
                    # Check if this is a rotatable bond (marked as 1 in conn_mat)
                    if self.bond_matrix[cur_atom, neighbor] == 1:
                        # IMPORTANT: Only process if (cur_atom, neighbor) is in branch_keys
                        # This ensures we follow the correct direction (parent -> child)
                        # and avoid processing the reverse direction
                        if (cur_atom, neighbor) in branch_keys:
                            # Create a copy of visited for this branch search
                            visited_copy = visited.copy()
                            # Find all atoms connected to neighbor (excluding cur_atom side)
                            connected_atoms = self.search_connected_atoms(self.bond_matrix, neighbor, visited_copy)
                            
                            if len(connected_atoms) > 0:
                                # Check if connected atoms contain at least one non-H atom
                                non_h_atoms = [atom for atom in connected_atoms if atom not in self.h_atom_indices]
                                
                                if len(non_h_atoms) > 0:  # Only consider as torsion if has non-H atoms
                                    torsions.append((cur_atom, neighbor))
                                    # Create mask for all connected atoms (including H for completeness)
                                    mask = np.zeros(self.atoms_num, dtype=bool)
                                    mask[connected_atoms] = True
                                    torsion_masks.append(mask)
        
        torsions = np.array(torsions)
        torsion_masks = np.array(torsion_masks) if len(torsion_masks) > 0 else np.empty((0, self.atoms_num), dtype=bool)

        return torsions, torsion_masks

    def _get_intra_pairs(self, skip_h_atoms: bool = True) -> np.ndarray:
        """Compute intra-molecular atom pairs for scoring.

        Intra pairs are atom pairs that:
        1. Are not both in root atoms (or connected to root).
        2. Are not within 3 bond distance.
        3. Are not in the same branch.
        4. Are filtered by hydrogen atom inclusion based on skip_h_atoms.

        Args:
            skip_h_atoms: Whether to skip hydrogen atoms. True gives default Vina
                behaviour; False includes hydrogen atoms (for AD4). Defaults to True.

        Returns:
            Intra pair indices, shape (N, 2).
        """
        def find_atoms_within_bonds(
            atom_idx: int, depth: int, connected_atoms: list[int]
        ) -> None:
            """Collect atoms reachable from atom_idx within depth bonds."""
            if atom_idx not in connected_atoms:
                connected_atoms.append(atom_idx)
                if depth > 0:
                    for neighbor in np.where(self.bond_matrix[atom_idx] > 0)[0]:
                        find_atoms_within_bonds(neighbor, depth - 1, connected_atoms)

        try:
            # Copy branch information dictionary
            branch_topology = copy.deepcopy(self.branch_atoms)
            parent_atoms = np.array([list(key) for key in branch_topology.keys()])[:, 0]
            child_atoms = np.array([list(key) for key in branch_topology.keys()])[:, 1]
            root_atoms = copy.deepcopy(self.root_atoms)

            # Get root atoms (add child atoms connected to root)
            for root_atom in self.root_atoms:
                if root_atom in parent_atoms:
                    child_indices = np.argwhere(parent_atoms == root_atom).flatten()
                    for idx in child_indices:
                        root_atoms.append(child_atoms[idx].item())

            # Get max branch atoms
            if len(branch_topology) > 0:
                max_branch_atoms = max(len(branch_topology[branch]) for branch in branch_topology) + 1  # +1 for parent atom
                num_branches = len(branch_topology.keys())

                # Get branch matrix
                branch_matrix = np.ones((num_branches, max_branch_atoms)) * -1
                for branch_idx, branch_key in enumerate(branch_topology):
                    branch_atoms = branch_topology[branch_key]
                    branch_matrix[branch_idx, 0:len(branch_atoms) + 1] = [branch_key[0]] + branch_atoms
            else:
                branch_matrix = np.array([])

            # Get intra_pairs_index
            intra_pairs_index = []
            n_atoms = len(self.ligand_coords)
            for i in range(n_atoms):
                # Skip hydrogen atoms if enabled
                if skip_h_atoms and i in self.h_atom_indices:
                    continue

                connected_atoms = []
                find_atoms_within_bonds(i, 3, connected_atoms)

                for j in range(i + 1, n_atoms):
                    # Skip hydrogen atoms if enabled
                    if skip_h_atoms and j in self.h_atom_indices:
                        continue

                    if any([
                        (i in root_atoms and j in root_atoms),  # skip pairs of root atoms (or connected to root)
                        j in connected_atoms,  # skip pairs of atoms within 3 bond distance
                    ]):
                        continue
                
                    # Check if in the same branch
                    if len(branch_matrix) > 0:
                        i_branch = np.where(branch_matrix == i)
                        j_branch = np.where(branch_matrix == j)
                        if (len(i_branch[0]) > 0 and len(j_branch[0]) > 0 and 
                            np.any(np.isin(i_branch[0], j_branch[0]))):
                            continue
                    
                    intra_pairs_index.append((i, j))
            
            intra_pairs_index = np.array(intra_pairs_index)

            return intra_pairs_index

        except Exception as e:
            self.logger.warning(f'[_get_intra_pairs] Error: {str(e)}')
            intra_pairs_index = np.array([], dtype=int).reshape(0, 2)

            return intra_pairs_index

    def save_pdbqt(
        self, output_path: str, coordinates: np.ndarray | None = None
    ) -> None:
        """Save ligand molecule to PDBQT file with structure information preserved.

        Preserves ROOT, BRANCH, ENDBRANCH, TORSDOF, and REMARK lines, which is
        essential for maintaining rotatable bond definitions and molecular topology.

        Args:
            output_path: Path to the output PDBQT file.
            coordinates: Coordinates to save, shape (self.atoms_num, 3).
                If None, uses current ligand coordinates. Defaults to None.

        Raises:
            ValueError: If coordinates shape does not match atom count.
        """
        # Use current coordinates if not provided
        if coordinates is None:
            coordinates = self.ligand_coords
        else:
            # Validate coordinates shape
            coordinates = np.asarray(coordinates)
            if coordinates.shape != (self.atoms_num, 3):
                msg = (
                    f"Coordinates shape mismatch: expected ({self.atoms_num}, 3), "
                    f"got {coordinates.shape}"
                )
                self.logger.error(msg)
                raise ValueError(msg)
        
        # Call ligand_parser.save_to_pdbqt with keep_structure=True to preserve
        # ROOT, BRANCH, ENDBRANCH, TORSDOF, and other structural information
        self.ligand_parser.save_to_pdbqt(
            output_path=output_path,
            keep_structure=True,
            coordinates=coordinates
        )
        
        # self.logger.info(f"Saved ligand to {output_path}")

    def __repr__(self) -> str:
        """Return string representation with atom count and torsion count."""
        return f"{self.__class__.__name__}({self.file_path}): {self.atoms_num} atoms, {len(self.torsions)} torsions."


class BaseProteinLoader(ABC):
    """Abstract base class for protein data loaders.

    Provides shared utilities for pocket extraction, flexible residue
    identification, chi-angle torsion computation, and intra-pair generation.
    Subclasses must implement the load_molecule() method.
    """

    def __init__(self, config: object) -> None:
        self.config = config
        self.logger = config.logger
        self.dtype = np.dtype(config.dtype)  # No actual use in data loading.

        self.load_molecule()

    @abstractmethod
    def load_molecule(self):
        raise NotImplementedError

    def _extract_receptor_pocket(
        self,
        pocket_center: list | np.ndarray,
        pocket_size: list | np.ndarray = [20, 20, 20],
        expand_size: float = 8,
        expand_residue: bool = True,
        include_hetatm: bool = False,
    ) -> tuple[list[int], np.ndarray]:
        """Extract receptor pocket atom indices based on a cubic box.

        Args:
            pocket_center: [cx, cy, cz], center of the pocket box.
            pocket_size: [sx, sy, sz], box edge lengths. Defaults to [20, 20, 20].
            expand_size: Extra margin added to half box size along each axis. Defaults to 8.
            expand_residue: If True, include all atoms of a residue when any ATOM in
                that residue falls inside the pocket. Defaults to True.
            include_hetatm: If True, keep HETATM atoms that fall inside the pocket.
                Defaults to False.

        Returns:
            A tuple of:
                - Sorted list of atom indices (0-based) belonging to the pocket.
                - Boolean mask of shape (N,) where N is the total number of atoms.
        """
        # Convert pocket center and size to numpy arrays
        pocket_center = np.array(pocket_center, dtype=float)
        pocket_size = np.array(pocket_size, dtype=float)

        # Vina energy threshold is 8.0A; when ligand atoms are at the edge
        # of the pocket, they should be expanded by 8A.
        expand_margin = float(expand_size)
        lower_bound = pocket_center - pocket_size / 2.0 - expand_margin
        upper_bound = pocket_center + pocket_size / 2.0 + expand_margin

        # Initial mask: atoms whose coordinates fall inside the expanded box
        protein_coords = self.protein_parser.get_coordinates()
        inside_mask = (
            (protein_coords[:, 0] >= lower_bound[0]) & (protein_coords[:, 0] <= upper_bound[0]) &
            (protein_coords[:, 1] >= lower_bound[1]) & (protein_coords[:, 1] <= upper_bound[1]) &
            (protein_coords[:, 2] >= lower_bound[2]) & (protein_coords[:, 2] <= upper_bound[2])
        )

        record_types = self.protein_parser.record_types
        res_names = self.protein_parser.res_names
        res_seqs = self.protein_parser.res_seqs
        chain_ids = self.protein_parser.chain_ids

        pocket_indices = set()

        # 1) Handle ATOM records
        if expand_residue:
            # Collect residue keys that have at least one atom inside the box
            selected_residues = set()
            for idx, inside in enumerate(inside_mask):
                if not inside:
                    continue
                if record_types[idx] != "ATOM":
                    continue
                key = (res_names[idx], res_seqs[idx], chain_ids[idx])
                selected_residues.add(key)

            # Add all atoms whose residue key is selected (only ATOM records)
            for idx in range(len(record_types)):
                if record_types[idx] != "ATOM":
                    continue
                key = (res_names[idx], res_seqs[idx], chain_ids[idx])
                if key in selected_residues:
                    pocket_indices.add(idx)
        else:
            # Do not expand residues, only keep ATOM atoms that are inside
            for idx, inside in enumerate(inside_mask):
                if not inside:
                    continue
                if record_types[idx] == "ATOM":
                    pocket_indices.add(idx)

        # 2) Handle HETATM records (per-atom selection)
        if include_hetatm:
            for idx, inside in enumerate(inside_mask):
                if not inside:
                    continue
                if record_types[idx] == "HETATM":
                    pocket_indices.add(idx)

        # Sort indices for deterministic order
        pocket_indices = sorted(pocket_indices)

        # Create boolean mask for efficient array operations
        pocket_mask = np.zeros(len(record_types), dtype=bool)
        pocket_mask[pocket_indices] = True

        return pocket_indices, pocket_mask

    def build_rdkit_to_pdbqt_mapping(
        self, rdkit_coords: np.ndarray, pdbqt_coords: np.ndarray, tol: float = 1e-4
    ) -> np.ndarray:
        """Build mapping from RDKit atom indices to PDBQT atom indices via coordinate matching.

        Uses a fast path if atoms are already in the same order; otherwise builds an
        explicit mapping using nearest-neighbor matching.

        Args:
            rdkit_coords: Coordinates from RDKit molecule, shape (N, 3).
            pdbqt_coords: Coordinates from PDBQT file (canonical order), shape (N, 3).
            tol: Tolerance for coordinate matching in Angstrom. Defaults to 1e-4.

        Returns:
            Mapping array of shape (N,) where mapping[rdkit_idx] = pdbqt_idx.
            Returns np.arange(N) when the fast path succeeds.

        Raises:
            ValueError: If atom counts differ or mapping cannot be established.
        """
        n_rdkit = rdkit_coords.shape[0]
        n_pdbqt = pdbqt_coords.shape[0]
        
        # Check atom count consistency
        if n_rdkit != n_pdbqt:
            msg = f"Atom count mismatch: RDKit has {n_rdkit} atoms, PDBQT has {n_pdbqt} atoms"
            self.logger.error(msg)
            raise ValueError(msg)
        
        # Fast path: check if atoms are already in the same order
        coord_diffs = rdkit_coords - pdbqt_coords
        max_diff = np.max(np.linalg.norm(coord_diffs, axis=1))
        
        if max_diff < tol:
            # All atoms match in order - use identity mapping
            return np.arange(n_rdkit, dtype=int)
        
        # Slow path: build explicit mapping using nearest neighbor matching
        self.logger.warning(
            f"RDKit and PDBQT atom order differs (max_diff={max_diff:.2e} >= tol={tol}). "
            "Building explicit coordinate-based mapping..."
        )
        
        # Compute pairwise distance matrix: dist[i, j] = distance between rdkit_i and pdbqt_j
        diffs = rdkit_coords[:, None, :] - pdbqt_coords[None, :, :]  # shape (N_rdkit, N_pdbqt, 3)
        dist_matrix = np.linalg.norm(diffs, axis=2)  # shape (N_rdkit, N_pdbqt)
        
        # For each RDKit atom, find closest PDBQT atom
        mapping = np.argmin(dist_matrix, axis=1)  # shape (N_rdkit,)
        min_dists = np.min(dist_matrix, axis=1)
        
        # Validate mapping quality
        # 1. Check that all matches are within tolerance
        if np.any(min_dists > tol):
            bad_indices = np.where(min_dists > tol)[0]
            msg = (
                f"Failed to match {len(bad_indices)} RDKit atoms within tolerance {tol}. "
                f"Examples: RDKit atom {bad_indices[0]} has min_dist={min_dists[bad_indices[0]]:.2e}"
            )
            self.logger.error(msg)
            raise ValueError(msg)
        
        # 2. Check that mapping is one-to-one (no duplicate assignments)
        unique_targets = np.unique(mapping)
        if len(unique_targets) != n_rdkit:
            msg = (
                f"Mapping is not one-to-one: {n_rdkit} RDKit atoms mapped to "
                f"{len(unique_targets)} unique PDBQT atoms. Possible coordinate ambiguity."
            )
            self.logger.error(msg)
            raise ValueError(msg)
        
        # Log statistics
        avg_diff = np.mean(min_dists)
        max_diff_actual = np.max(min_dists)
        self.logger.info(
            f"Built RDKit->PDBQT mapping: avg_diff={avg_diff:.2e}, max_diff={max_diff_actual:.2e}"
        )
        
        return mapping

    def _identify_flexible_residues(
        self, flexible_residues: str | None = None
    ) -> tuple[list[tuple[str, str, str]], np.ndarray]:
        """Identify flexible residues in the pocket.

        Args:
            flexible_residues: Residue specification string in the format
                "A:123,A:125,B:45,..." (chain:resSeq). If None, residues are
                automatically detected based on distance threshold. Defaults to None.

        Returns:
            A tuple of:
                - flexible_residue_keys: [(chain_id, res_seq, res_name), ...],
                  sorted by (chain_id, res_seq).
                - flexible_residue_min_distances: Minimum distance of each flexible
                  residue to the pocket center, shape (F,).
        """
        atoms_num = self.atoms_num
        res_names = self.protein_parser.res_names
        res_seqs = self.protein_parser.res_seqs
        chain_ids = self.protein_parser.chain_ids
        record_types = self.protein_parser.record_types
        
        # Only consider ATOM records for flexible residues
        atom_mask = np.array([rt == "ATOM" for rt in record_types], dtype=bool)
        
        if flexible_residues is None:
            # Automatic detection based on distance threshold
            pocket_coords = self.protein_parser.get_coordinates()
            distances = np.linalg.norm(pocket_coords - np.array(self.box_center), axis=1)
            
            # Find residues with at least one heavy atom within threshold
            candidate_residues = set()
            for idx in range(atoms_num):
                if not atom_mask[idx]:
                    continue
                # Skip H atoms when checking distance
                if idx in self.h_atom_indices:
                    continue
                if distances[idx] <= self.flexible_residue_search_radius:
                    res_name = res_names[idx]
                    res_seq = res_seqs[idx]
                    chain_id = chain_ids[idx]
                    
                    # Skip non-flexible residue types
                    if res_name in NON_FLEXIBLE_RESIDUES:
                        continue
                    
                    # Check if it's a standard residue
                    if res_name not in RESTYPE_NAME_TO_ATOM14_NAMES:
                        self.logger.warning(
                            f"Residue {chain_id}:{res_seq}:{res_name} is not a standard residue, skipping"
                        )
                        continue
                    
                    # Check chain ID is not empty
                    if not chain_id or chain_id.strip() == "":
                        msg = f"Residue {res_name}:{res_seq} has empty chain ID, cannot be flexible"
                        self.logger.error(msg)
                        raise ValueError(msg)
                    
                    candidate_residues.add((chain_id, res_seq, res_name))
            
            flexible_residue_keys = list(candidate_residues)

        else:
            # Parse user-specified flexible residues
            flexible_residue_keys = []
            seen_residues = set()  # Track already processed residues to detect duplicates
            
            # Build a map of available residues in the pocket
            available_residues = {}
            for idx in range(len(res_names)):
                if not atom_mask[idx]:
                    continue
                res_name = res_names[idx]
                res_seq = res_seqs[idx]
                chain_id = chain_ids[idx]
                key = (chain_id, res_seq)
                available_residues[key] = res_name
            
            # Parse input string: "A:123,A:125,B:45,..."
            parts = [p.strip() for p in flexible_residues.split(",") if p.strip()]
            
            for part in parts:
                if ":" not in part:
                    msg = f"Invalid flexible residue format: {part}, expected 'Chain:ResSeq'"
                    self.logger.error(msg)
                    raise ValueError(msg)
                
                chain_id, res_seq = part.split(":", 1)
                chain_id = chain_id.strip()
                res_seq = res_seq.strip()  # Parse residue sequence number (keep as string)
                
                # Check chain ID is not empty
                if not chain_id:
                    msg = f"Chain ID cannot be empty in flexible residue specification: {part}"
                    self.logger.error(msg)
                    raise ValueError(msg)
                
                # Check for duplicate specification
                key = (chain_id, res_seq)
                if key in seen_residues:
                    msg = (
                        f"Duplicate flexible residue specification: {chain_id}:{res_seq} "
                        f"appears more than once in the input"
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
                seen_residues.add(key)
                
                # Check if residue exists in pocket
                if key not in available_residues:
                    msg = (
                        f"Specified flexible residue {chain_id}:{res_seq} not found in pocket. "
                        f"Make sure it's within the pocket region."
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
                
                res_name = available_residues[key]
                
                # Check if it's a non-flexible residue type
                if res_name in NON_FLEXIBLE_RESIDUES:
                    msg = (
                        f"Residue {chain_id}:{res_seq}:{res_name} is {res_name}, "
                        f"which cannot be flexible (GLY/ALA/PRO not allowed)"
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
                
                # Check if it's a standard residue
                if res_name not in RESTYPE_NAME_TO_ATOM14_NAMES:
                    msg = (
                        f"Residue {chain_id}:{res_seq}:{res_name} is not a standard residue "
                        f"defined in atom14 representation"
                    )
                    self.logger.error(msg)
                    raise ValueError(msg)
                
                flexible_residue_keys.append((chain_id, res_seq, res_name))
        
        # Sort by (chain_id, res_seq)
        flexible_residue_keys.sort(key=lambda x: (x[0], int(x[1]) if x[1].lstrip('-').isdigit() else x[1]))

        # Calculate minimum distance of each flexible residue to pocket center (excluding H atoms)
        pocket_coords = self.protein_parser.get_coordinates()
        pocket_center = np.array(self.box_center)
        
        flexible_residue_min_distances = []
        for chain_id, res_seq, res_name in flexible_residue_keys:
            # Find all heavy atoms belonging to this residue
            residue_atom_indices = []
            for idx in range(len(res_names)):
                if not atom_mask[idx]:
                    continue
                if chain_ids[idx] == chain_id and res_seqs[idx] == res_seq:
                    # Only include heavy atoms for distance calculation
                    if idx not in self.h_atom_indices:
                        residue_atom_indices.append(idx)
            
            if len(residue_atom_indices) == 0:
                # This should not happen if validation passed
                msg = f"No heavy atoms found for residue {chain_id}:{res_seq}:{res_name}"
                self.logger.error(msg)
                raise ValueError(msg)
                
            # Calculate minimum distance using only heavy atoms
            residue_coords = pocket_coords[residue_atom_indices]
            distances = np.linalg.norm(residue_coords - pocket_center, axis=1)
            min_distance = np.min(distances)
            flexible_residue_min_distances.append(min_distance)
        
        flexible_residue_min_distances = np.array(flexible_residue_min_distances, dtype=float)

        return flexible_residue_keys, flexible_residue_min_distances

    @abstractmethod
    def _separate_flex_rigid_components(self) -> None:
        raise NotImplementedError

    def _get_torsions_and_masks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get torsions and torsion masks for flexible residues based on chi angle definitions.

        Returns:
            A tuple of:
                - flex_torsions: Torsion atom pairs in flex_coords indices, shape (F, 4, 2).
                  Padding value (-1, -1) for residues with fewer than 4 chi angles.
                - flex_torsion_masks: Boolean masks for affected atoms, shape (F, 4, max_m).
                  True means the atom is rotated by this torsion (includes H atoms).
                - flex_movable_heavy_mask: Boolean mask for movable heavy atoms,
                  shape (F, max_m). True means the atom is affected by at least one
                  chi angle (side chain atoms); False for fixed atoms or padding.
        """
        if not hasattr(self, 'flexible_residue_keys') or len(self.flexible_residue_keys) == 0:
            # No flexible residues
            return np.empty((0, 4, 2), dtype=int), np.empty((0, 4, 0), dtype=bool), np.empty((0, 0), dtype=bool)
        
        F = len(self.flexible_residue_keys)
        max_m = self.flex_coords.shape[1]
        
        # Initialize torsion arrays with padding
        flex_torsions = np.full((F, 4, 2), -1, dtype=int)  # -1 for padding
        flex_torsion_masks = np.zeros((F, 4, max_m), dtype=bool)  # False for padding
        
        # Get protein parser data for atom name lookup
        res_names = self.protein_parser.res_names
        res_seqs = self.protein_parser.res_seqs
        chain_ids = self.protein_parser.chain_ids
        atom_names = self.protein_parser.atom_names
        
        # Process each flexible residue
        for flex_idx, (chain_id, res_seq, res_name) in enumerate(self.flexible_residue_keys):
            # Build mapping from atom14 names to local indices in flex_coords
            atom14_names = RESTYPE_NAME_TO_ATOM14_NAMES[res_name]
            atom14_to_local_idx = {}  # {atom14_name: local_idx in flex_coords[flex_idx]}
            local_to_atom14_idx = {}  # {local_idx: atom14_idx}
            
            # Get atom indices for this residue from flex_atom_indices
            residue_atom_indices = self.flex_atom_indices[flex_idx]  # shape [max_m]
            
            # Count valid atoms in this residue
            num_valid_atoms = np.sum(residue_atom_indices >= 0)
            
            for local_idx, global_idx in enumerate(residue_atom_indices):
                if global_idx == -1:  # Padding position
                    continue
                
                # Get atom name from parser
                atom_name = atom_names[global_idx]
                
                # Find position in atom14 representation
                if atom_name in atom14_names:
                    atom14_idx = atom14_names.index(atom_name)
                    atom14_to_local_idx[atom_name] = local_idx
                    local_to_atom14_idx[local_idx] = atom14_idx
            
            # Build local bond matrix for this residue
            # Extract submatrix from global bond_matrix
            local_bond_matrix = np.zeros((num_valid_atoms, num_valid_atoms), dtype=int)
            valid_global_indices = residue_atom_indices[:num_valid_atoms]
            
            for i in range(num_valid_atoms):
                for j in range(num_valid_atoms):
                    global_i = valid_global_indices[i]
                    global_j = valid_global_indices[j]
                    local_bond_matrix[i, j] = self.bond_matrix[global_i, global_j]
            
            # Build H atom attachment mapping (which heavy atom each H is bonded to)
            h_to_heavy_map = {}  # {h_local_idx: heavy_local_idx}
            
            for local_idx in range(num_valid_atoms):
                global_idx = valid_global_indices[local_idx]
                
                # Check if this is a hydrogen atom
                if global_idx in self.h_atom_indices:
                    # Find which heavy atom it's bonded to
                    for neighbor_local_idx in range(num_valid_atoms):
                        if local_bond_matrix[local_idx, neighbor_local_idx] > 0:
                            neighbor_global_idx = valid_global_indices[neighbor_local_idx]
                            if neighbor_global_idx not in self.h_atom_indices:
                                h_to_heavy_map[local_idx] = neighbor_local_idx
                                break
            
            # Get chi angle definitions for this residue type
            chi_angles_defs = CHI_ANGLES_ATOM14_INDICES.get(res_name, [])
            
            # Process each chi angle and calculate torsion masks
            for chi_idx, chi_def in enumerate(chi_angles_defs):
                if chi_idx >= 4:  # Only support up to 4 chi angles
                    break
                
                # chi_def is [i, j, k, l] - atom14 indices defining the dihedral angle
                # The torsion bond is between atoms j and k
                i_atom14, j_atom14, k_atom14, l_atom14 = chi_def
                
                # Get atom14 names for j and k (the torsion bond atoms)
                j_name = atom14_names[j_atom14]
                k_name = atom14_names[k_atom14]
                
                # Check if both torsion atoms exist in this residue
                if j_name not in atom14_to_local_idx or k_name not in atom14_to_local_idx:
                    self.logger.warning(
                        f"Residue {chain_id}:{res_seq}:{res_name} chi{chi_idx+1} is incomplete: "
                        f"torsion atoms {j_name} or {k_name} not found. Skipping this torsion."
                    )
                    continue
                
                # Get local indices for the torsion bond
                j_local = atom14_to_local_idx[j_name]
                k_local = atom14_to_local_idx[k_name]
                
                # Store torsion pair (parent, child) where rotation happens around j->k bond
                flex_torsions[flex_idx, chi_idx, 0] = j_local
                flex_torsions[flex_idx, chi_idx, 1] = k_local
                
                # Calculate torsion mask - atoms affected by this rotation
                # All atoms on the "k side" of the j-k bond will be rotated
                # We use atom14 order to determine which atoms are on the k side
                
                # Atoms with atom14_idx > k_atom14 are typically on the k side
                # This follows the atom14 naming convention where atoms are ordered
                # from backbone to side chain tip
                affected_heavy_atoms = set()
                
                for local_idx, atom14_idx in local_to_atom14_idx.items():
                    # An atom is affected if it comes after k in the atom14 order
                    # AND it's not on the j side of the bond
                    if atom14_idx > k_atom14:
                        affected_heavy_atoms.add(local_idx)
                
                # Build torsion mask including H atoms
                for heavy_local_idx in affected_heavy_atoms:
                    flex_torsion_masks[flex_idx, chi_idx, heavy_local_idx] = True
                    
                    # Also include H atoms bonded to this heavy atom
                    for h_local_idx, bonded_heavy_idx in h_to_heavy_map.items():
                        if bonded_heavy_idx == heavy_local_idx:
                            flex_torsion_masks[flex_idx, chi_idx, h_local_idx] = True
        
        # Compute movable atom mask: atoms affected by ANY chi angle
        flex_movable_mask = np.any(flex_torsion_masks, axis=1) 
        flex_movable_heavy_mask = flex_movable_mask & self.flex_heavy_atom_mask
        
        return flex_torsions, flex_torsion_masks, flex_movable_heavy_mask

    def _compute_initial_intra_pairs(
        self, cutoff: float = 10.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute initial intra atom pairs for flex-flex and flex-rigid interactions.

        - flex-flex pairs: between movable heavy atoms in one flexible residue and
          heavy atoms in other flexible residues.
        - flex-rigid pairs: between movable heavy atoms in flexible residues and
          nearby rigid heavy atoms within a distance cutoff.

        Args:
            cutoff: Distance cutoff in Angstrom for flex-rigid pairs. Defaults to 10.0.

        Returns:
            A tuple of:
                - flex_flex_pair_indices: shape (K1, 4), integer indices
                  [flex_res_idx_A, flex_local_idx_A, flex_res_idx_B, flex_local_idx_B].
                - flex_rigid_pair_indices: shape (K2, 3), integer indices
                  [flex_res_idx, flex_local_idx, rigid_idx].
        """
        # Preconditions: pocket must be partitioned
        if not hasattr(self, 'flex_coords') or not hasattr(self, 'rigid_coords'):
            msg = "Pocket must be partitioned before computing pairs. Call partition_pocket()."
            self.logger.error(msg)
            raise ValueError(msg)

        # Ensure heavy movable mask is available
        if not hasattr(self, 'flex_movable_heavy_mask'):
            msg = "Missing flex_movable_heavy_mask. Please call _get_torsions_and_masks() first."
            self.logger.error(msg)
            raise ValueError(msg)

        F = self.flex_coords.shape[0]
        if self.flex_movable_heavy_mask.shape[0] != F:
            msg = (
                f"Shape mismatch: flex_movable_heavy_mask has {self.flex_movable_heavy_mask.shape[0]} residues, "
                f"but flex_coords has {F}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        max_m = self.flex_coords.shape[1]

        cutoff_sq = float(cutoff) * float(cutoff)

        # -------- Compute flex-flex pairs --------
        flex_flex_pairs_set = set()
        if F > 0:
            for f in range(F):
                movable_mask_f = self.flex_movable_heavy_mask[f]
                local_indices_f = np.nonzero(movable_mask_f)[0]

                if len(local_indices_f) == 0:
                    continue

                for local_idx_f in local_indices_f:
                    coord_f = self.flex_coords[f, local_idx_f]

                    for g in range(F):
                        if g == f:
                            continue
                        # Effective heavy atoms in residue g
                        heavy_mask_g = self.flex_heavy_atom_mask[g]
                        heavy_local_indices_g = np.nonzero(heavy_mask_g)[0]
                        if len(heavy_local_indices_g) == 0:
                            continue

                        coords_g = self.flex_coords[g, heavy_local_indices_g]  # [Ng, 3]
                        diffs = coords_g - coord_f  # [Ng, 3]
                        d2 = np.einsum('ij,ij->i', diffs, diffs)  # [Ng]
                        near_idx = np.nonzero(d2 <= cutoff_sq)[0]
                        if len(near_idx) == 0:
                            continue

                        for idx in near_idx:
                            local_idx_g = int(heavy_local_indices_g[idx])
                            a = (int(f), int(local_idx_f))
                            b = (int(g), int(local_idx_g))
                            # Canonical ordering to deduplicate
                            key = (a, b) if a <= b else (b, a)
                            flex_flex_pairs_set.add(key)

        if len(flex_flex_pairs_set) == 0:
            flex_flex_pair_indices = np.empty((0, 4), dtype=int)
        else:
            # Convert set of ((f, li), (g, lj)) to array [f, li, g, lj]
            flex_flex_pair_indices = np.array([[a[0], a[1], b[0], b[1]] for (a, b) in sorted(list(flex_flex_pairs_set))], dtype=int)

        # -------- Compute flex-rigid pairs --------
        rigid_heavy_mask = self.rigid_heavy_atom_mask
        rigid_heavy_indices = np.nonzero(rigid_heavy_mask)[0]
        rigid_heavy_coords = self.rigid_coords[rigid_heavy_mask]

        if F == 0 or len(rigid_heavy_indices) == 0:
            flex_rigid_pair_indices = np.empty((0, 3), dtype=int)
        else:
            all_pairs = []
            for f in range(F):
                movable_mask = self.flex_movable_heavy_mask[f]
                local_indices = np.nonzero(movable_mask)[0]

                for local_idx in local_indices:
                    flex_coord = self.flex_coords[f, local_idx]
                    diffs = rigid_heavy_coords - flex_coord  # [R_h, 3]
                    d2 = np.einsum('ij,ij->i', diffs, diffs)
                    near_mask = d2 <= cutoff_sq
                    if np.any(near_mask):
                        near_rigid = rigid_heavy_indices[near_mask]
                        for rigid_idx in near_rigid:
                            all_pairs.append((int(f), int(local_idx), int(rigid_idx)))

            if len(all_pairs) == 0:
                flex_rigid_pair_indices = np.empty((0, 3), dtype=int)
            else:
                flex_rigid_pair_indices = np.array(all_pairs, dtype=int)

        return flex_flex_pair_indices, flex_rigid_pair_indices

    def partition_pocket(
        self, flex_dock: bool = True, flexible_residues: str | None = None
    ) -> None:
        """Partition pocket atoms into flexible and rigid components.

        Performs the following steps:
        1. Identify flexible residues (auto-detect or user-specified).
        2. Separate pocket atoms into flex/rigid components with padding.
        3. Calculate chi-angle torsions and torsion masks for flexible residues.

        After completion, the following attributes are set:
            - flexible_residue_keys: [(chain_id, res_seq, res_name), ...].
            - flexible_residue_min_distances: shape (F,), min distance to pocket center.
            - flex_coords: shape (F, max_m, 3), flexible residue coordinates (padding: 0.0).
            - flex_xs_types: shape (F, max_m), XS types (-1 for padding/H).
            - flex_xs_vdw_radii: shape (F, max_m), VDW radii (0.0 for padding/H).
            - flex_atom_indices: shape (F, max_m), original pocket indices (-1 for padding).
            - flex_torsions: shape (F, 4, 2), torsion atom pairs ((-1, -1) for padding).
            - flex_torsion_masks: shape (F, 4, max_m), boolean masks for rotated atoms (includes H).
            - flex_movable_heavy_mask: shape (F, max_m), boolean mask for movable side chain heavy atoms.
            - rigid_coords: shape (R, 3), rigid atom coordinates.
            - rigid_xs_types: shape (R), XS types (-1 for H).
            - rigid_xs_vdw_radii: shape (R), VDW radii.

        Args:
            flex_dock: Whether to perform flexible docking. If False, all pocket
                atoms are treated as rigid. Defaults to True.
            flexible_residues: Residue specification string in the format
                "A:123,A:125,B:45,..." (chain:resSeq). If None, residues are
                automatically detected within flexible_residue_search_radius.
                Defaults to None.
        """
        # Identify flexible residues
        if flex_dock:
            self.flexible_residue_keys, self.flexible_residue_min_distances = self._identify_flexible_residues(flexible_residues)
        else:
            # Rigid docking: no flexible residues
            self.flexible_residue_keys = []
            self.flexible_residue_min_distances = np.array([], dtype=float)
        
        # Separate flex and rigid components
        self._separate_flex_rigid_components()

        # Get torsions and masks
        self.flex_torsions, self.flex_torsion_masks, self.flex_movable_heavy_mask = self._get_torsions_and_masks()
        self.flex_flex_pair_indices, self.flex_rigid_pair_indices = self._compute_initial_intra_pairs()
    
    def save_pdbqt(
        self, output_path: str, flex_coordinates: np.ndarray | None = None
    ) -> None:
        """Save protein pocket to PDBQT file with updated flexible residue coordinates.

        If flexible residue coordinates are provided, they are first mapped back to the
        original pocket atom indices before saving.

        Args:
            output_path: Path to the output PDBQT file.
            flex_coordinates: Flexible residue coordinates, shape (F, max_m, 3).
                If None, uses current self.flex_coords. Defaults to None.

        Raises:
            ValueError: If flex_coordinates shape does not match expected dimensions,
                or if the pocket has not been partitioned yet.
        """
        # Check if pocket has been partitioned
        if not hasattr(self, 'flex_coords') or not hasattr(self, 'flex_atom_indices'):
            msg = "Pocket must be partitioned before saving. Call partition_pocket() first."
            self.logger.error(msg)
            raise ValueError(msg)
        
        # Use current flex coordinates if not provided
        if flex_coordinates is None:
            flex_coordinates = self.flex_coords
        else:
            # Validate flex_coordinates shape
            flex_coordinates = np.asarray(flex_coordinates)
            expected_shape = self.flex_coords.shape
            if flex_coordinates.shape != expected_shape:
                msg = (
                    f"Flex coordinates shape mismatch: expected {expected_shape}, "
                    f"got {flex_coordinates.shape}"
                )
                self.logger.error(msg)
                raise ValueError(msg)
        
        # Create a copy of current pocket coordinates
        updated_pocket_coords = self.protein_coords.copy()
        
        # Update flexible residue coordinates back to pocket coordinates
        F = flex_coordinates.shape[0]
        max_m = flex_coordinates.shape[1]
        
        for flex_idx in range(F):
            for local_idx in range(max_m):
                global_idx = self.flex_atom_indices[flex_idx, local_idx]
                
                # Skip padding positions (marked as -1)
                if global_idx == -1:
                    continue
                
                # Update the coordinate in pocket
                updated_pocket_coords[global_idx] = flex_coordinates[flex_idx, local_idx]
        
        # Call protein_parser.save_to_pdbqt with keep_structure=False
        # This saves only atom coordinates without ROOT/BRANCH structure info
        self.protein_parser.save_to_pdbqt(
            output_path=output_path,
            keep_structure=False,
            coordinates=updated_pocket_coords
        )
        
        # self.logger.info(f"Saved protein pocket to {output_path}")
        
    def __repr__(self) -> str:
        """Return string representation with pocket summary information."""
        lines = []
        lines.append(f"{self.__class__.__name__}({self.protein_file_path}):")
        lines.append(f"  Pocket center: [{self.box_center[0]:.2f}, {self.box_center[1]:.2f}, {self.box_center[2]:.2f}],")
        lines.append(f"  Pocket size: [{self.box_size[0]:.2f}, {self.box_size[1]:.2f}, {self.box_size[2]:.2f}],")

        # Atom count information
        total_atoms = self.atoms_num
        flex_atoms = 0
        rigid_atoms = 0
        
        if hasattr(self, 'flex_coords'):
            # Count non-padding flex atoms
            flex_atoms = np.sum(self.flex_atom_indices >= 0)
        
        if hasattr(self, 'rigid_coords'):
            rigid_atoms = len(self.rigid_coords)

        if hasattr(self, 'flex_rigid_pair_indices'):
            flex_rigid_pairs = len(self.flex_rigid_pair_indices)
        
        lines.append(f"  Total atoms: {total_atoms}, Flexible: {flex_atoms}, Rigid: {rigid_atoms},")
        
        # Flexible residue information
        if hasattr(self, 'flexible_residue_keys') and len(self.flexible_residue_keys) > 0:
            n_flex_residues = len(self.flexible_residue_keys)
            lines.append(f"  Flexible residues: {n_flex_residues}:")
            
            for chain_id, res_seq, res_name in self.flexible_residue_keys:
                lines.append(f"    - {res_name}:{res_seq}:{chain_id}")
        else:
            lines.append(f"  Flexible residues: 0 (rigid docking)")
        
        return "\n".join(lines)
        