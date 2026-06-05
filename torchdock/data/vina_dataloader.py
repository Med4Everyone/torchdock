"""
Vina/Vinardo data loaders for molecular docking.

This module implements ligand and protein data loaders specialized for
the AutoDock Vina and Vinardo scoring functions. It handles PDBQT file
parsing, atom type conversion, torsion extraction, and pocket filtering.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import os
import uuid

import numpy as np
from openbabel import openbabel
from rdkit import Chem

from torchdock.data.base_dataloader import BaseLigandLoader, BaseProteinLoader
from torchdock.data.pdbqtParser import PdbqtParser


class VinaLigandLoader(BaseLigandLoader):
    """Ligand data loader for Vina and Vinardo scoring functions.

    Parses PDBQT ligand files, converts atom types (AD -> element -> XS),
    extracts torsion trees, and prepares all data structures required for
    Vina/Vinardo scoring and conformation search.
    """

    def __init__(self, config: object) -> None:
        self.file_path = config.ligand_file_path
        self.score_function = config.score_function

        super().__init__(config)

    def _validate_file_and_score_function(self) -> None:
        """Validate ligand file existence, format, and score function compatibility.

        Raises:
            FileNotFoundError: If the ligand PDBQT file does not exist.
            ValueError: If the file extension is not .pdbqt, or if the score
                function is not one of 'vina' or 'vinardo'.
        """
        # Check if the file exists
        if not os.path.exists(self.file_path):
            msg = f"The file {self.file_path} does not exist."
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        # Validate file extension
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext != ".pdbqt":
            msg = "Ligand file must be PDBQT (.pdbqt) for VinaLigandLoader"
            self.logger.error(msg)
            raise ValueError(msg)

        # Check score function
        allowed_score_functions = ["vina", "vinardo"]
        if self.score_function not in allowed_score_functions:
            msg = f"Score function must be one of {allowed_score_functions}"
            self.logger.error(msg)
            raise ValueError(msg)

    def load_molecule(self) -> None:
        """Load ligand molecule data from PDBQT file.

        This method performs the following steps:
        1. Validates file existence and format
        2. Parses PDBQT file to extract atom coordinates and types
        3. Validates coordinates (NaN/Inf check, duplicate atom check)
        4. Calculates molecular geometric center and center atom
        5. Converts PDBQT to mol2 via OpenBabel and loads with RDKit
        6. Builds atom mapping between RDKit and PDBQT order
        7. Constructs bond connectivity matrix from RDKit bonds
        8. Validates molecular connectivity (single connected component)
        9. Converts atom types (AD type -> element type -> XS type)
        10. Identifies rotatable bonds and updates connectivity matrix
        11. Calculates torsions, torsion masks, and intra-molecular pairs
        12. Prepares atomic numbers and adjacency matrix for RMSD calculation

        Sets:
            self.ligand_parser: PdbqtParser, PDBQT file parser object
            self.atoms_num: int, number of atoms in the ligand
            self.ad_type_strs: list of str, AutoDock atom type strings from PDBQT
            self.ligand_coords: np.ndarray, shape (N, 3), atom coordinates in Angstrom
            self.position_center_coords: np.ndarray, shape (3,), geometric center of molecule
            self.center_atom: int, index of atom closest to geometric center
            self.atom_center_coords: np.ndarray, shape (3,), coordinates of center atom
            self.rdkit_to_pdbqt_mapping: np.ndarray, shape (N,), mapping[rdkit_idx] = pdbqt_idx
            self.bond_matrix: np.ndarray, shape (N, N), connectivity matrix
                - bond_matrix[i][j] = 2: normal bond
                - bond_matrix[i][j] = 1: rotatable bond
                - bond_matrix[i][j] = 0: no bond
            self.ad_types: np.ndarray, shape (N,), AutoDock4 atom type indices
            self.el_types: np.ndarray, shape (N,), element type indices
            self.xs_types: np.ndarray, shape (N,), X-Score atom type indices
            self.h_atom_indices: set of int, indices of hydrogen atoms
            self.root_atoms: list of int, root atom indices from PDBQT ROOT section
            self.branch_atoms: dict, {(parent, child): [child_branch_atoms]}
            self.xs_vdw_radii: np.ndarray, shape (N,), van der Waals radii for XS types
            self.torsions: np.ndarray, shape (M, 2), torsion atom pairs (parent, child)
            self.torsion_masks: np.ndarray, shape (M, N), boolean masks for atoms in each torsion branch
            self.degrees_of_freedom: float, molecular degrees of freedom for docking
            self.intra_pairs_index: np.ndarray, shape (K, 2), intra-molecular atom pairs for energy calculation
            self.atomicnums: np.ndarray, shape (N,), atomic numbers in PDBQT order
            self.adjacency_matrix: np.ndarray, shape (N, N), adjacency matrix for RMSD (1=bonded, 0=not bonded)

        Raises:
            FileNotFoundError: If PDBQT file does not exist.
            ValueError: If file format is invalid, coordinates contain NaN/Inf,
                duplicate atoms exist, molecule is disconnected, or RDKit conversion fails.
        """
        from torchdock.constants.vina_constants import (
            EL_TYPE_H,
            XS_VDW_RADIIS,
            ad_type_to_el_type,
            el_type_to_xs_type,
            str_to_ad_type,
        )
        from torchdock.constants.vinardo_constants import VINARDO_XS_VDW_RADII

        # Validate file and score function
        self._validate_file_and_score_function()

        # Parse the PDBQT file and extract coordinates, atom types and other data
        self.ligand_parser = PdbqtParser(self.file_path)

        self.atoms_num = len(self.ligand_parser)
        self.ad_type_strs = self.ligand_parser.ad_types
        self.ligand_coords = self.ligand_parser.get_coordinates()

        # Validate ligand coordinates
        self._validate_ligand_coordinates(self.ligand_coords, self.atoms_num)

        # Calculate the molecular geometric center and the coordinates of the central atom.
        self.position_center_coords = np.mean(self.ligand_coords, axis=0)

        dists = np.linalg.norm(self.ligand_coords - self.position_center_coords, axis=1)
        self.center_atom = np.argmin(dists)
        self.atom_center_coords = self.ligand_coords[self.center_atom]

        # Use OpenBabel to convert pdbqt file to mol2 file, and use rdkit to load the mol2 file.
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("pdbqt", "mol2")
        obmol = openbabel.OBMol()
        obConversion.ReadFile(obmol, self.file_path)
        tmp_mol2_path = os.path.join(os.path.dirname(self.file_path), "tmp_" + str(uuid.uuid4()) + ".mol2")
        obConversion.WriteFile(obmol, tmp_mol2_path)

        rdkit_mol = Chem.MolFromMol2File(tmp_mol2_path, sanitize=False) # ignore valence check
        if rdkit_mol is None:
            os.remove(tmp_mol2_path)
            msg = "Error: RdKit was unable to read mol2 file"
            self.logger.error(msg)
            raise ValueError(msg)
        os.remove(tmp_mol2_path)

        # Build mapping from RDKit atom indices to PDBQT atom indices
        rdkit_coords = rdkit_mol.GetConformer().GetPositions()
        self.rdkit_to_pdbqt_mapping = self.build_rdkit_to_pdbqt_mapping(rdkit_coords, self.ligand_coords)

        # Build connectivity matrix using the mapping to align with PDBQT atom order
        # bond_matrix[i][j] = 2 means atoms i and j are connected by a normal bond
        # bond_matrix[i][j] = 1 will be set later for rotatable bonds (in BRANCH sections)
        self.bond_matrix = np.zeros((self.atoms_num, self.atoms_num), dtype=int)
        
        for bond in rdkit_mol.GetBonds():
            rdkit_i = bond.GetBeginAtomIdx()
            rdkit_j = bond.GetEndAtomIdx()
            
            # Map RDKit atom indices to PDBQT atom indices
            pdbqt_i = self.rdkit_to_pdbqt_mapping[rdkit_i]
            pdbqt_j = self.rdkit_to_pdbqt_mapping[rdkit_j]
            
            # Set bond in connectivity matrix (symmetric)
            self.bond_matrix[pdbqt_i, pdbqt_j] = 2
            self.bond_matrix[pdbqt_j, pdbqt_i] = 2
        
        # Check ligand connectivity to ensure it's a single molecule
        self._check_ligand_connectivity(self.bond_matrix)

        # Initialize type arrays with default values
        self.ad_types = np.full(self.atoms_num, -1, dtype=int) # AutoDock4 atom types
        self.el_types = np.full(self.atoms_num, -1, dtype=int) # Element types
        self.xs_types = np.full(self.atoms_num, -1, dtype=int) # X-Score atom types

        # Convert AD type strings to AD types, xs_types will be automatically filled for non-standard metals (Cu, Na, K, etc.)
        str_to_ad_type(self.ad_type_strs, self.ad_types, xs_types=self.xs_types, validate=True)

        # Convert AD types to element types
        ad_type_to_el_type(self.ad_types, self.el_types)

        # Get the atomic indices of H atoms
        self.h_atom_indices = set(np.where(self.el_types == EL_TYPE_H)[0])

        # Get the root and branch atoms
        self.root_atoms, self.branch_atoms = self.ligand_parser.get_root_branch_from_pdbqt()

        # Update connectivity matrix for branch atoms (rotatable bonds)
        for branch in self.branch_atoms:
            atom1, atom2 = branch
            
            # Validate that the bond exists as a normal bond (value=2) before marking as rotatable
            if self.bond_matrix[atom1, atom2] != 2:
                self.logger.warning(f"Rotatable bond ({atom1}, {atom2}) is not at a normal bond position. ")
            
            # Mark as rotatable bond (value=1)
            self.bond_matrix[atom1, atom2] = 1
            self.bond_matrix[atom2, atom1] = 1

        # Convert element types to X-Score types
        el_type_to_xs_type(self.ad_types, self.el_types, self.xs_types, self.bond_matrix)

        # Get X-Score van der Waals radii
        if self.score_function == "vina":
            self.xs_vdw_radii = XS_VDW_RADIIS[self.xs_types]
        elif self.score_function == "vinardo":
            self.xs_vdw_radii = VINARDO_XS_VDW_RADII[self.xs_types]
            

        # Get torsions and torsion masks
        self.torsions, self.torsion_masks = self._get_torsions_and_masks()

        # Calculate degrees of freedom
        if len(self.torsions) == 0:
            self.degrees_of_freedom = 0
        else:
            self.degrees_of_freedom = (len(self.branch_atoms) + len(self.torsions)) / 2

        # Get intra pair for ligand intra energy calculation
        self.intra_pairs_index = self._get_intra_pairs()

        # Get atomic numbers from RDKit and map to PDBQT atom order for RMSD calculation
        rdkit_atomicnums = np.array([atom.GetAtomicNum() for atom in rdkit_mol.GetAtoms()], dtype=int)
        self.atomicnums = rdkit_atomicnums[self.rdkit_to_pdbqt_mapping]
        
        # Build adjacency matrix for RMSD calculation from bond_matrix
        self.adjacency_matrix = (self.bond_matrix > 0).astype(int)


class VinaProteinLoader(BaseProteinLoader):
    """Protein data loader for Vina and Vinardo scoring functions.

    Parses PDBQT protein files, extracts pocket atoms within the docking box,
    converts atom types, and separates flexible/rigid residue components
    for Vina/Vinardo scoring.
    """

    def __init__(self, config: object) -> None:
        self.protein_file_path = config.protein_file_path
        self.box_center = config.box_center
        self.box_size = config.box_size
        self.score_function = config.score_function

        self.include_hetatm = config.include_hetatm
        self.flexible_residue_search_radius = config.flexible_residue_search_radius

        super().__init__(config)

    def _validate_file_and_score_function(self) -> None:
        """Validate protein file existence, format, score function, and box parameters.

        Raises:
            FileNotFoundError: If the protein PDBQT file does not exist.
            ValueError: If the file extension is not .pdbqt, the score function
                is not one of 'vina' or 'vinardo', or box center/size are invalid.
        """
        # Check if the file exists
        if not os.path.exists(self.protein_file_path):
            msg = f"The file {self.protein_file_path} does not exist."
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        # Validate file extension
        ext = os.path.splitext(self.protein_file_path)[1].lower()
        if ext != ".pdbqt":
            msg = "Protein file must be PDBQT (.pdbqt) for VinaProteinLoader"
            self.logger.error(msg)
            raise ValueError(msg)

        # Check score function
        allowed_score_functions = ["vina", "vinardo"]
        if self.score_function not in allowed_score_functions:
            msg = f"Score function must be one of {allowed_score_functions}"
            self.logger.error(msg)
            raise ValueError(msg)

        # Validate box center and size
        if not isinstance(self.box_center, (np.ndarray, list)) or len(self.box_center) != 3:
            msg = "Box center must be a numpy array or list of shape (3,)"
            self.logger.error(msg)
            raise ValueError(msg)

        if not isinstance(self.box_size, (np.ndarray, list)) or len(self.box_size) != 3:
            msg = "Box size must be a numpy array or list of shape (3,)"
            self.logger.error(msg)
            raise ValueError(msg)
          
    def load_molecule(self) -> None:
        """Load protein molecule data from PDBQT file.

        Parses the protein PDBQT file, extracts pocket atoms within the docking
        box, builds atom type mappings (AD -> element -> XS), constructs the
        bond connectivity matrix, and assigns van der Waals radii.

        Sets:
            self.protein_parser: PdbqtParser, PDBQT file parser object
            self.pocket_atom_indices: np.ndarray, indices of pocket atoms
            self.pocket_atom_mask: np.ndarray, boolean mask for pocket atoms
            self.atoms_num: int, number of pocket atoms
            self.ad_type_strs: list of str, AutoDock atom type strings
            self.protein_coords: np.ndarray, shape (N, 3), pocket atom coordinates
            self.rdkit_to_pdbqt_mapping: np.ndarray, mapping from RDKit to PDBQT indices
            self.bond_matrix: np.ndarray, shape (N, N), connectivity matrix
            self.ad_types: np.ndarray, shape (N,), AutoDock4 atom type indices
            self.el_types: np.ndarray, shape (N,), element type indices
            self.xs_types: np.ndarray, shape (N,), X-Score atom type indices
            self.h_atom_indices: set of int, indices of hydrogen atoms
            self.xs_vdw_radii: np.ndarray, shape (N,), van der Waals radii for XS types

        Raises:
            FileNotFoundError: If protein PDBQT file does not exist.
            ValueError: If file format is invalid or RDKit conversion fails.
        """
        from torchdock.constants.vina_constants import (
            EL_TYPE_H,
            XS_VDW_RADIIS,
            ad_type_to_el_type,
            el_type_to_xs_type,
            str_to_ad_type,
        )
        from torchdock.constants.vinardo_constants import VINARDO_XS_VDW_RADII

        # Validate file and score function
        self._validate_file_and_score_function()

        # Parse the PDBQT file and extract coordinates, atom types and other data
        self.protein_parser = PdbqtParser(self.protein_file_path)

        # Filter protein parser to include only pocket atoms
        self.pocket_atom_indices, self.pocket_atom_mask = self._extract_receptor_pocket(self.box_center, self.box_size, include_hetatm=self.include_hetatm)
        self.protein_parser.filter_atoms(self.pocket_atom_mask)

        self.atoms_num = len(self.protein_parser)
        self.ad_type_strs = self.protein_parser.ad_types
        self.protein_coords = self.protein_parser.get_coordinates()

        # Save pocket coordinates to a temporary PDB file
        tmp_pdb_path = os.path.join(os.path.dirname(self.protein_file_path), "tmp_" + str(uuid.uuid4()) + ".pdb")
        self.protein_parser.save_to_pdb(tmp_pdb_path)

        # Use rdkit to read PDB file
        rdkit_mol = Chem.MolFromPDBFile(tmp_pdb_path, sanitize=False)
        if rdkit_mol is None:
            os.remove(tmp_pdb_path)
            msg = "Error: RdKit was unable to read PDB file"
            self.logger.error(msg)
            raise ValueError(msg)
        os.remove(tmp_pdb_path)

        # Build mapping from RDKit atom indices to PDBQT atom indices
        rdkit_coords = rdkit_mol.GetConformer().GetPositions()
        self.rdkit_to_pdbqt_mapping = self.build_rdkit_to_pdbqt_mapping(rdkit_coords, self.protein_coords)

        # Build connectivity matrix using the mapping to align with PDBQT atom order
        self.bond_matrix = np.zeros((self.atoms_num, self.atoms_num), dtype=int)
        
        for bond in rdkit_mol.GetBonds():
            rdkit_i = bond.GetBeginAtomIdx()
            rdkit_j = bond.GetEndAtomIdx()
            
            # Map RDKit atom indices to PDBQT atom indices
            pdbqt_i = self.rdkit_to_pdbqt_mapping[rdkit_i]
            pdbqt_j = self.rdkit_to_pdbqt_mapping[rdkit_j]
            
            # Set bond in connectivity matrix (symmetric)
            self.bond_matrix[pdbqt_i, pdbqt_j] = 2
            self.bond_matrix[pdbqt_j, pdbqt_i] = 2
        
        # Initialize atom type arrays
        self.ad_types = np.full(self.atoms_num, -1, dtype=int) # AutoDock4 atom types
        self.el_types = np.full(self.atoms_num, -1, dtype=int) # Element types
        self.xs_types = np.full(self.atoms_num, -1, dtype=int) # X-Score atom types

        # Convert AD type strings to AD types, xs_types will be automatically filled for non-standard metals (Cu, Na, K, etc.)
        str_to_ad_type(self.ad_type_strs, self.ad_types, xs_types=self.xs_types, validate=True)

        # Convert AD types to element types
        ad_type_to_el_type(self.ad_types, self.el_types)

        # Get the atomic indices of H atoms
        self.h_atom_indices = set(np.where(self.el_types == EL_TYPE_H)[0]) 

        # Convert element types to X-Score types
        el_type_to_xs_type(self.ad_types, self.el_types, self.xs_types, self.bond_matrix)

        # Get X-Score van der Waals radii
        if self.score_function == "vina":
            self.xs_vdw_radii = XS_VDW_RADIIS[self.xs_types]
        elif self.score_function == "vinardo":
            self.xs_vdw_radii = VINARDO_XS_VDW_RADII[self.xs_types]

    def _separate_flex_rigid_components(self) -> None:
        """Separate pocket atoms into flexible and rigid components.

        This method organizes pocket atoms into two groups:
        1. Flexible components: atoms belonging to flexible residues
           - Organized as [F, max_m, 3] where F is number of flexible residues,
             max_m is maximum atoms per residue
           - Padded positions use -1 for xs_types and indices, 0.0 for vdw_radii

        2. Rigid components: all remaining pocket atoms
           - Organized as [R, 3] where R is number of rigid atoms
        
        Sets:
            self.flex_coords: np.ndarray, shape [F, max_m, 3], flexible residue coordinates (padding: 0.0)
            self.flex_xs_types: np.ndarray, shape [F, max_m], XS types for flex atoms (-1 for padding/H)
            self.flex_xs_vdw_radii: np.ndarray, shape [F, max_m], VDW radii for flex atoms (0.0 for padding/H)
            self.flex_atom_indices: np.ndarray, shape [F, max_m], original pocket indices (-1 for padding)
            self.flex_heavy_atom_mask: np.ndarray, shape [F, max_m], boolean mask for heavy atoms (False for H/padding)
            
            self.rigid_coords: np.ndarray, shape [R, 3], rigid atom coordinates
            self.rigid_xs_types: np.ndarray, shape [R], XS types for rigid atoms (-1 for H)
            self.rigid_xs_vdw_radii: np.ndarray, shape [R], VDW radii for rigid atoms
            self.rigid_heavy_atom_mask: np.ndarray, shape [R], boolean mask for heavy atoms (False for H)
        """
        if not hasattr(self, 'flexible_residue_keys') or len(self.flexible_residue_keys) == 0:           
            self.flex_coords = np.empty((0, 0, 3), dtype=float)
            self.flex_xs_types = np.empty((0, 0), dtype=int)
            self.flex_xs_vdw_radii = np.empty((0, 0), dtype=float)
            self.flex_atom_indices = np.empty((0, 0), dtype=int)
            self.flex_heavy_atom_mask = np.empty((0, 0), dtype=bool)
            
            self.rigid_coords = self.protein_coords.copy()
            self.rigid_xs_types = self.xs_types.copy()
            self.rigid_xs_vdw_radii = self.xs_vdw_radii.copy()
            self.rigid_heavy_atom_mask = (self.xs_types >= 0)
            
            return
        
        # Build mapping from (chain_id, res_seq) to flexible residue index
        flex_residue_map = {}
        for flex_idx, (chain_id, res_seq, res_name) in enumerate(self.flexible_residue_keys):
            flex_residue_map[(chain_id, res_seq)] = flex_idx
        
        # Get pocket atom information
        res_names = self.protein_parser.res_names
        res_seqs = self.protein_parser.res_seqs
        chain_ids = self.protein_parser.chain_ids
        record_types = self.protein_parser.record_types
        
        # Classify atoms into flexible residues
        F = len(self.flexible_residue_keys)
        flex_residue_atom_lists = [[] for _ in range(F)]  # [F] lists of atom indices
        rigid_atom_list = []  # List of atom indices for rigid atoms
        
        for atom_idx in range(self.atoms_num):
            # Only consider ATOM records for flexible residues
            if record_types[atom_idx] != "ATOM":
                rigid_atom_list.append(atom_idx)
                continue
            
            chain_id = chain_ids[atom_idx]
            res_seq = res_seqs[atom_idx]
            key = (chain_id, res_seq)
            
            if key in flex_residue_map:
                flex_idx = flex_residue_map[key]
                flex_residue_atom_lists[flex_idx].append(atom_idx)
            else:
                rigid_atom_list.append(atom_idx)
        
        # Determine max_m (maximum number of atoms in any flexible residue)
        max_m = max(len(atom_list) for atom_list in flex_residue_atom_lists) if F > 0 else 0
        
        # Initialize flex arrays with padding
        self.flex_coords = np.zeros((F, max_m, 3), dtype=float)  # Padding: 0.0 (won't be used due to xs_type=-1)
        self.flex_xs_types = np.full((F, max_m), -1, dtype=int)  # -1 for padding (same as H atoms)
        self.flex_xs_vdw_radii = np.zeros((F, max_m), dtype=float)  # 0.0 for padding
        self.flex_atom_indices = np.full((F, max_m), -1, dtype=int)  # -1 for padding positions
        
        # Fill flex arrays
        for flex_idx in range(F):
            atom_list = flex_residue_atom_lists[flex_idx]
            
            for local_idx, atom_idx in enumerate(atom_list):
                self.flex_coords[flex_idx, local_idx] = self.protein_coords[atom_idx]
                self.flex_xs_types[flex_idx, local_idx] = self.xs_types[atom_idx]
                self.flex_xs_vdw_radii[flex_idx, local_idx] = self.xs_vdw_radii[atom_idx]
                self.flex_atom_indices[flex_idx, local_idx] = atom_idx
        
        # Build rigid arrays
        R = len(rigid_atom_list)
        self.rigid_coords = np.zeros((R, 3), dtype=float)
        self.rigid_xs_types = np.zeros(R, dtype=int)
        self.rigid_xs_vdw_radii = np.zeros(R, dtype=float)
        
        for rigid_idx, atom_idx in enumerate(rigid_atom_list):
            self.rigid_coords[rigid_idx] = self.protein_coords[atom_idx]
            self.rigid_xs_types[rigid_idx] = self.xs_types[atom_idx]
            self.rigid_xs_vdw_radii[rigid_idx] = self.xs_vdw_radii[atom_idx]
        
        # Compute heavy atom masks for flexible and rigid components
        # Heavy atoms have xs_types >= 0 (H atoms and padding have xs_types = -1)
        self.flex_heavy_atom_mask = (self.flex_xs_types >= 0)
        self.rigid_heavy_atom_mask = (self.rigid_xs_types >= 0)
