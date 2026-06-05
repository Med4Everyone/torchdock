"""
Amino acid atom14 representation constants for protein structure processing.

Defines the mapping from residue types to their 14-atom naming convention,
chi-angle atom index definitions, and the set of non-flexible residues
used throughout the TorchDock flexible docking pipeline.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

# Maps each amino acid residue type to its 14-atom naming convention (Atom14 format), with empty strings for unused positions.
RESTYPE_NAME_TO_ATOM14_NAMES = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
}

# Maps each residue type to a list of χ-angle definitions as 4-tuples of Atom14 indices (empty if no rotatable side-chain angles).
CHI_ANGLES_ATOM14_INDICES = {
    'ALA': [],
    'ARG': [[0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    'ASN': [[0, 1, 4, 5], [1, 4, 5, 6]],
    'ASP': [[0, 1, 4, 5], [1, 4, 5, 6]],
    'CYS': [[0, 1, 4, 5]],
    'GLN': [[0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7]],
    'GLU': [[0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7]],
    'GLY': [],
    'HIS': [[0, 1, 4, 5], [1, 4, 5, 6]],
    'ILE': [[0, 1, 4, 5], [1, 4, 5, 7]],
    'LEU': [[0, 1, 4, 5], [1, 4, 5, 6]], 
    'LYS': [[0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
    'MET': [[0, 1, 4, 5], [1, 4, 5, 6], [4, 5, 6, 7]],
    'PHE': [[0, 1, 4, 5], [1, 4, 5, 6]],
    'PRO': [[0, 1, 4, 5], [1, 4, 5, 6]],
    'SER': [[0, 1, 4, 5]],
    'THR': [[0, 1, 4, 5]],
    'TRP': [[0, 1, 4, 5], [1, 4, 5, 6]],
    'TYR': [[0, 1, 4, 5], [1, 4, 5, 6]],
    'VAL': [[0, 1, 4, 5]],
}

# Residues with no meaningful side-chain flexibility 
NON_FLEXIBLE_RESIDUES = {
    "ALA",  # Alanine – side chain is a symmetric methyl group; rotation around Cα–Cβ has no energetic or structural consequence.
    "GLY",  # Glycine – no side chain (only a hydrogen atom); therefore, no χ dihedral angle exists.
    "PRO",  # Proline – side chain is cyclized back to the backbone amide nitrogen, forming a rigid pyrrolidine ring that severely restricts χ-angle flexibility.
} 
