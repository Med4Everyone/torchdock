"""
PDBQT file format parser.

This module provides the PdbqtParser class for reading, manipulating,
and writing PDBQT files used in AutoDock-based molecular docking workflows.
It supports parsing atom records, preserving structural information
(ROOT/BRANCH hierarchy), filtering atoms, and exporting to PDBQT or PDB format.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import numpy as np


class PdbqtParser:
    """
    Parser for PDBQT file format.
    Each instance represents a single PDBQT file.
    """
    
    # PDBQT field column positions (using Python slice notation)
    RECORD_TYPE_COLS = (0, 6)      # Record type: ATOM or HETATM, 1-6
    SERIAL_COLS = (6, 11)          # Atom serial number, 7-11
    ATOM_NAME_COLS = (12, 16)      # Atom name, 13-16
    RES_NAME_COLS = (17, 20)       # Residue name, 18-20
    CHAIN_ID_COLS = (21, 22)       # Chain identifier, 22
    RES_SEQ_COLS = (22, 26)        # Residue sequence number, 23-26
    ICODE_COLS = (26, 27)          # Insertion code, 27
    X_COORD_COLS = (30, 38)        # X coordinate, 31-38
    Y_COORD_COLS = (38, 46)        # Y coordinate, 39-46
    Z_COORD_COLS = (46, 54)        # Z coordinate, 47-54
    OCCUPANCY_COLS = (54, 60)      # Occupancy, 55-60
    TEMP_FACTOR_COLS = (60, 66)    # Temperature factor, 61-66
    CHARGE_COLS = (70, 76)         # PDBQT charge, 71-76
    AD_TYPE_START = (76, 79)       # AutoDock type, 77-79
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
        # Record metadata
        self.record_types = []  # ATOM or HETATM
        self.serials = []  # atom serial numbers (int)
        
        # Atom identification
        self.atom_names = []  # atom names
        self.res_names = []  # residue names
        self.chain_ids = []  # chain identifiers (can be empty string)
        self.res_seqs = []  # residue sequence numbers (can be empty string)
        self.icodes = []  # insertion codes (can be empty string)
        
        # Coordinates
        self.x_coords = []  # x coordinates (float)
        self.y_coords = []  # y coordinates (float)
        self.z_coords = []  # z coordinates (float)
        
        # Additional fields
        self.occupancies = []  # occupancy values (float, default 0.0)
        self.temp_factors = []  # temperature factors (float, default 0.0)
        self.charges = []  # PDBQT charges (float or None)
        self.ad_types = []  # AutoDock atom types (string)
        
        # Structural information preservation
        self.header_lines = []  # lines before first ATOM/HETATM (REMARK, ROOT, etc.)
        self.footer_lines = []  # lines after last ATOM/HETATM (TORSDOF, etc.)
        self.non_atom_lines = {}  # mapping of line index to non-ATOM lines (BRANCH, ENDBRANCH, etc.)
        
        # Original atom lines preservation (key: atom index 0-based, value: original line content)
        self.original_atom_lines = {}
        
        # Parse the file
        self._parse()
    
    def _parse(self):
        """Parse the PDBQT file and populate all field lists."""
        atom_count = 0
        first_atom_seen = False
        all_lines = []
        
        with open(self.file_path, 'r') as f:
            all_lines = f.readlines()
        
        for line_num, line in enumerate(all_lines, start=1):
            # Check if this is an ATOM or HETATM record
            record_type = line[self.RECORD_TYPE_COLS[0]:self.RECORD_TYPE_COLS[1]].strip()
            
            if record_type not in ['ATOM', 'HETATM']:
                # Non-ATOM lines handling - always preserve during parsing
                if not first_atom_seen:
                    # Before first atom (REMARK, ROOT, etc.)
                    self.header_lines.append(line)
                elif atom_count > 0:
                    # Between or after atoms (BRANCH, ENDBRANCH, TORSDOF, etc.)
                    self.non_atom_lines[atom_count - 1] = self.non_atom_lines.get(atom_count - 1, []) + [line]
                continue
            
            first_atom_seen = True
            
            try:
                # Record type (required)
                self.record_types.append(record_type)
                
                # Serial number (required, must be int)
                serial_str = line[self.SERIAL_COLS[0]:self.SERIAL_COLS[1]].strip()
                if not serial_str:
                    raise ValueError(f"Serial number is empty")
                self.serials.append(int(serial_str))
                
                # Atom name (required)
                atom_name = line[self.ATOM_NAME_COLS[0]:self.ATOM_NAME_COLS[1]].strip()
                if not atom_name:
                    raise ValueError(f"Atom name is empty")
                self.atom_names.append(atom_name)
                
                # Residue name (required)
                res_name = line[self.RES_NAME_COLS[0]:self.RES_NAME_COLS[1]].strip()
                if not res_name:
                    raise ValueError(f"Residue name is empty")
                self.res_names.append(res_name)
                
                # Chain ID (can be empty)
                chain_id = line[self.CHAIN_ID_COLS[0]:self.CHAIN_ID_COLS[1]]
                self.chain_ids.append(chain_id if chain_id.strip() else '')
                
                # Residue sequence number (can be empty, keep as string)
                res_seq = line[self.RES_SEQ_COLS[0]:self.RES_SEQ_COLS[1]].strip()
                self.res_seqs.append(res_seq)
                
                # Insertion code (can be empty)
                icode = line[self.ICODE_COLS[0]:self.ICODE_COLS[1]]
                self.icodes.append(icode if icode.strip() else '')
                
                # Coordinates (required, must be float)
                x_str = line[self.X_COORD_COLS[0]:self.X_COORD_COLS[1]].strip()
                y_str = line[self.Y_COORD_COLS[0]:self.Y_COORD_COLS[1]].strip()
                z_str = line[self.Z_COORD_COLS[0]:self.Z_COORD_COLS[1]].strip()
                
                if not x_str:
                    raise ValueError(f"X coordinate is empty")
                if not y_str:
                    raise ValueError(f"Y coordinate is empty")
                if not z_str:
                    raise ValueError(f"Z coordinate is empty")
                
                x = float(x_str)
                y = float(y_str)
                z = float(z_str)
                self.x_coords.append(x)
                self.y_coords.append(y)
                self.z_coords.append(z)
                
                # Occupancy (optional, default 0.0)
                occupancy_str = line[self.OCCUPANCY_COLS[0]:self.OCCUPANCY_COLS[1]].strip()
                occupancy = float(occupancy_str) if occupancy_str else 0.0
                self.occupancies.append(occupancy)
                
                # Temperature factor (optional, default 0.0)
                temp_factor_str = line[self.TEMP_FACTOR_COLS[0]:self.TEMP_FACTOR_COLS[1]].strip()
                temp_factor = float(temp_factor_str) if temp_factor_str else 0.0
                self.temp_factors.append(temp_factor)
                
                # PDBQT charge (optional, default None)
                charge_str = line[self.CHARGE_COLS[0]:self.CHARGE_COLS[1]].strip() if len(line) > self.CHARGE_COLS[0] else ''
                charge = float(charge_str) if charge_str else None
                self.charges.append(charge)
                
                # AD type (required)
                ad_type = line[self.AD_TYPE_START[0]:self.AD_TYPE_START[1]].strip()
                if not ad_type:
                    raise ValueError(f"AD type is empty")
                self.ad_types.append(ad_type)
                
                # Save original atom line (key: current atom index, value: original line without trailing newline)
                self.original_atom_lines[atom_count] = line.rstrip('\n')
                
                atom_count += 1
                
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"Error parsing line {line_num} in file '{self.file_path}': {e}\n"
                    f"Line content: {line.rstrip()}"
                ) from e
        
        # Collect footer lines (lines after last atom)
        if atom_count > 0:
            # Find remaining lines after the last atom line
            last_atom_line_idx = -1
            for idx, line in enumerate(all_lines):
                record_type = line[self.RECORD_TYPE_COLS[0]:self.RECORD_TYPE_COLS[1]].strip()
                if record_type in ['ATOM', 'HETATM']:
                    last_atom_line_idx = idx
            
            if last_atom_line_idx >= 0 and last_atom_line_idx < len(all_lines) - 1:
                # Check if these lines were already added to non_atom_lines
                remaining_lines = all_lines[last_atom_line_idx + 1:]
                stored_in_non_atom = self.non_atom_lines.get(atom_count - 1, [])
                
                # Only add to footer if not already in non_atom_lines
                for line in remaining_lines:
                    if line not in stored_in_non_atom:
                        self.footer_lines.append(line)
      
        # Validate that all field lists have the same length
        self._validate_field_lengths()
    
    def _validate_field_lengths(self):
        """Validate that all required field lists have the same length.

        Raises:
            ValueError: If any field length inconsistency is found.
        """
        expected_length = len(self.serials)
        
        # Check all required fields (core atom data fields only)
        fields_to_check = [
            ('record_types', self.record_types),
            ('atom_names', self.atom_names),
            ('res_names', self.res_names),
            ('chain_ids', self.chain_ids),
            ('res_seqs', self.res_seqs),
            ('icodes', self.icodes),
            ('x_coords', self.x_coords),
            ('y_coords', self.y_coords),
            ('z_coords', self.z_coords),
            ('occupancies', self.occupancies),
            ('temp_factors', self.temp_factors),
            ('charges', self.charges),
            ('ad_types', self.ad_types)
        ]
        
        for field_name, field_list in fields_to_check:
            if len(field_list) != expected_length:
                raise ValueError(
                    f"Field length mismatch in file '{self.file_path}': "
                    f"{field_name} has {len(field_list)} elements, "
                    f"expected {expected_length} (based on serials count)"
                )
    
    def __len__(self):
        """Return the number of atoms in the PDBQT file."""
        return len(self.serials)
    
    def get_atom(self, index: int) -> dict:
        """Get all fields for a specific atom by index.

        Args:
            index: Atom index (0-based).

        Returns:
            Dictionary containing all fields for the specified atom.
        """
        if index < 0 or index >= len(self):
            raise IndexError(f"Atom index {index} out of range [0, {len(self)})")
        
        return {
            'record_type': self.record_types[index],
            'serial': self.serials[index],
            'atom_name': self.atom_names[index],
            'res_name': self.res_names[index],
            'chain_id': self.chain_ids[index],
            'res_seq': self.res_seqs[index],
            'icode': self.icodes[index],
            'x': self.x_coords[index],
            'y': self.y_coords[index],
            'z': self.z_coords[index],
            'occupancy': self.occupancies[index],
            'temp_factor': self.temp_factors[index],
            'charge': self.charges[index],
            'ad_type': self.ad_types[index]
        }
    
    def get_coordinates(self) -> np.ndarray:
        """Get all atom coordinates as a numpy array.

        Returns:
            Array of shape (n_atoms, 3) containing (x, y, z) coordinates.
        """
        return np.array([self.x_coords, self.y_coords, self.z_coords]).T
    
    def filter_atoms(self, select_atom_mask: np.ndarray) -> None:
        """Filter atoms based on boolean mask, updating all field lists in-place.

        All instance field lists (record_types, serials, atom_names, coordinates,
        etc.) are updated to contain only the selected atoms.

        Args:
            select_atom_mask: Boolean mask array of shape (n_atoms,).
                True indicates atoms to keep.

        Raises:
            ValueError: If mask shape does not match atom count, or if validation fails.
        """
        # Validate mask shape
        if len(select_atom_mask) != len(self.serials):
            raise ValueError(
                f"Mask length ({len(select_atom_mask)}) does not match "
                f"atom count ({len(self.serials)})"
            )
        
        # Convert mask to numpy array if needed
        select_atom_mask = np.asarray(select_atom_mask, dtype=bool)
        
        # Filter all field lists using numpy boolean indexing for efficiency
        self.record_types = list(np.array(self.record_types)[select_atom_mask])
        self.serials = list(np.array(self.serials)[select_atom_mask])
        self.atom_names = list(np.array(self.atom_names)[select_atom_mask])
        self.res_names = list(np.array(self.res_names)[select_atom_mask])
        self.chain_ids = list(np.array(self.chain_ids)[select_atom_mask])
        self.res_seqs = list(np.array(self.res_seqs)[select_atom_mask])
        self.icodes = list(np.array(self.icodes)[select_atom_mask])
        
        # Coordinates
        self.x_coords = list(np.array(self.x_coords)[select_atom_mask])
        self.y_coords = list(np.array(self.y_coords)[select_atom_mask])
        self.z_coords = list(np.array(self.z_coords)[select_atom_mask])
        
        # Additional fields
        self.occupancies = list(np.array(self.occupancies)[select_atom_mask])
        self.temp_factors = list(np.array(self.temp_factors)[select_atom_mask])
        self.charges = list(np.array(self.charges, dtype=object)[select_atom_mask])
        self.ad_types = list(np.array(self.ad_types)[select_atom_mask])
        
        # Update original_atom_lines mapping after filtering
        old_indices = np.where(select_atom_mask)[0]
        new_original_lines = {}
        for new_idx, old_idx in enumerate(old_indices):
            if old_idx in self.original_atom_lines:
                new_original_lines[new_idx] = self.original_atom_lines[old_idx]
        self.original_atom_lines = new_original_lines
        
        # Validate field lengths after filtering
        self._validate_field_lengths()
    
    def get_root_branch_from_pdbqt(self) -> tuple[list[int], dict[tuple[int, int], list[int]]]:
        """Extract ROOT atoms and rotatable bonds from a ligand PDBQT file.

        Returns:
            Tuple of (root_atoms, branch_atoms):
                - root_atoms: List of atom indices (0-based) in the ROOT region.
                - branch_atoms: Dict mapping (atom_i, atom_j) to list of atom
                  indices in that branch, where (atom_i, atom_j) defines the
                  rotatable bond.

        Raises:
            ValueError: If HETATM records are found (not expected in ligand PDBQT).
        """
        # Validate that this is a ligand PDBQT file (only ATOM records, no HETATM)
        if 'HETATM' in self.record_types:
            hetatm_count = self.record_types.count('HETATM')
            raise ValueError(
                f"Found {hetatm_count} HETATM record(s) in ligand PDBQT file '{self.file_path}'. "
                "Ligand PDBQT files should only contain ATOM records. "
                "HETATM records are not expected."
            )
        
        # Build mapping from PDBQT atom serial number to 0-indexed atom index
        serial_to_index = {serial: idx for idx, serial in enumerate(self.serials)}
        
        # Parse ROOT and BRANCH sections from the original file
        root_atoms = []
        branch_atoms = {}
        
        with open(self.file_path, 'r') as f:
            lines = [x.rstrip('\n') for x in f.readlines()]
        
        # Track current context using a stack
        # Each element is: ('ROOT', None) or ('BRANCH', (atom_i, atom_j))
        context_stack = []
        
        for line in lines:
            line_type = line.split()[0] if line.strip() else ''
            
            if line_type == "ROOT":
                context_stack.append(('ROOT', None))
            
            elif line_type == "ENDROOT":
                if context_stack and context_stack[-1][0] == 'ROOT':
                    context_stack.pop()
            
            elif line_type == "BRANCH":
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        serial_i = int(parts[1])
                        serial_j = int(parts[2])
                        if serial_i in serial_to_index and serial_j in serial_to_index:
                            atom_i = serial_to_index[serial_i]
                            atom_j = serial_to_index[serial_j]
                            branch_atoms[(atom_i, atom_j)] = []
                            context_stack.append(('BRANCH', (atom_i, atom_j)))
                    except ValueError:
                        pass
            
            elif line_type == "ENDBRANCH":
                if context_stack and context_stack[-1][0] == 'BRANCH':
                    context_stack.pop()
            
            elif line.startswith("ATOM  "):
                # Only process ATOM records, not HETATM
                try:
                    serial_num = int(line[self.SERIAL_COLS[0]:self.SERIAL_COLS[1]].strip())
                    if serial_num in serial_to_index:
                        atom_idx = serial_to_index[serial_num]
                        
                        # Add atom to the most recent context (ROOT or BRANCH)
                        if context_stack:
                            context_type, context_data = context_stack[-1]
                            if context_type == 'ROOT':
                                root_atoms.append(atom_idx)
                            elif context_type == 'BRANCH':
                                branch_atoms[context_data].append(atom_idx)
                except (ValueError, IndexError):
                    pass
        
        return root_atoms, branch_atoms

    def save_to_pdbqt(self, output_path: str, keep_structure: bool = False,
                      coordinates: np.ndarray | None = None) -> None:
        """Save current PDBQT data to a specified file.

        Args:
            output_path: Path to the output PDBQT file.
            keep_structure: If True, preserves structural information (REMARK,
                ROOT, BRANCH, ENDBRANCH, TORSDOF, etc.) for ligand molecules.
                If False, only writes atom records for receptor proteins.
                Defaults to False.
            coordinates: Coordinate matrix of shape (n_atoms, 3). If provided,
                these coordinates will be written instead of the current ones.
                Defaults to None.

        Raises:
            ValueError: If field validation fails or coordinates shape is invalid.
        """
        # Validate coordinates if provided
        if coordinates is not None:
            coordinates = np.asarray(coordinates)
            
            # Validate coordinates shape
            if coordinates.shape != (len(self), 3):
                raise ValueError(
                    f"Coordinates shape {coordinates.shape} does not match "
                    f"expected shape ({len(self)}, 3)"
                )
        
        # Validate field lengths before saving
        self._validate_field_lengths()
        
        with open(output_path, 'w') as f:
            # Write header lines if requested (REMARK, ROOT, etc.)
            if keep_structure:
                for line in self.header_lines:
                    f.write(line if line.endswith('\n') else line + '\n')
            
            # Write atom records - use original line format and replace coordinates
            for i in range(len(self)):
                # Get the atom line with updated coordinates
                line = self._get_atom_line_with_coords(i, coordinates)
                f.write(line + '\n')
                
                # Write non-atom lines after this atom (BRANCH, ENDBRANCH, etc.)
                if keep_structure and i in self.non_atom_lines:
                    for non_atom_line in self.non_atom_lines[i]:
                        f.write(non_atom_line if non_atom_line.endswith('\n') else non_atom_line + '\n')
            
            # Write footer lines if requested (TORSDOF, etc.)
            if keep_structure:
                for line in self.footer_lines:
                    f.write(line if line.endswith('\n') else line + '\n')
    
    def _get_atom_line_with_coords(self, index: int,
                                   coordinates: np.ndarray | None = None) -> str:
        """Get the original atom line with coordinates replaced.

        Uses the original line format from the parsed file and only replaces
        the coordinate fields (columns 31-54), preserving all other formatting.

        Args:
            index: Atom index (0-based).
            coordinates: Coordinate matrix of shape (n_atoms, 3). If provided,
                uses coordinates[index] for this atom. Defaults to None.

        Returns:
            Atom line with updated coordinates (without newline).
        """
        # Get original line
        original_line = self.original_atom_lines.get(index, '')
        
        if not original_line:
            raise ValueError(f"No original line found for atom index {index}")
        
        # Get coordinates to use
        if coordinates is not None:
            x = coordinates[index, 0]
            y = coordinates[index, 1]
            z = coordinates[index, 2]
        else:
            x = self.x_coords[index]
            y = self.y_coords[index]
            z = self.z_coords[index]
        
        # Format coordinates with 8.3f format (columns 31-38, 39-46, 47-54, 0-indexed: 30-38, 38-46, 46-54)
        x_str = f"{x:8.3f}"
        y_str = f"{y:8.3f}"
        z_str = f"{z:8.3f}"
        
        # Replace coordinate fields in original line
        # Ensure line is long enough
        line = original_line.ljust(80)
        
        # Replace coordinates (columns 31-54, 0-indexed: 30-54)
        line = line[:30] + x_str + y_str + z_str + line[54:]
        
        return line.rstrip()

    def save_to_pdb(self, output_path: str, coordinates: np.ndarray | None = None) -> None:
        """Save current PDBQT data to a standard PDB file.

        Converts PDBQT format to standard PDB by removing PDBQT-specific
        structure information (ROOT, BRANCH, ENDBRANCH, TORSDOF) and AutoDock
        atom types. Element symbol and segment ID fields are left blank.

        Args:
            output_path: Path to the output PDB file.
            coordinates: Coordinate matrix of shape (n_atoms, 3). If provided,
                these coordinates will be written instead of the current ones.
                Defaults to None.

        Raises:
            ValueError: If field validation fails or coordinates shape is invalid.

        Note:
            Element symbols are left blank (columns 77-78) as RDKit can infer
            them from atom names. Segment identifiers (columns 73-76) and PDB
            charge field (columns 79-80) are also left blank.
        """
        # Validate coordinates if provided
        if coordinates is not None:
            coordinates = np.asarray(coordinates)
            
            # Validate coordinates shape
            if coordinates.shape != (len(self), 3):
                raise ValueError(
                    f"Coordinates shape {coordinates.shape} does not match "
                    f"expected shape ({len(self)}, 3)"
                )
        
        # Validate field lengths before saving
        self._validate_field_lengths()
        
        with open(output_path, 'w') as f:
            # Write atom records - use original line format with updated coordinates
            # PDB and PDBQT share the same format for columns 1-66, only differ after
            for i in range(len(self)):
                # Get the atom line with updated coordinates, then truncate to PDB format
                line = self._get_atom_line_with_coords(i, coordinates)
                # For PDB format, keep columns 1-66 (0-indexed: 0-66) and pad with spaces
                # PDB format ends at column 80 with element symbol and charge fields
                pdb_line = line[:66].ljust(80)
                f.write(pdb_line.rstrip() + '\n')

    def __repr__(self):
        return f"PdbqtParser('{self.file_path}', {len(self)} atoms)"
