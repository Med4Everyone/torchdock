"""
AutoDock-Vina atom type constants and X-Score type definitions.

Ported from AutoDock-Vina/src/lib/atom_constants.h. Provides AD type,
element type, and X-Score type enumerations, atom kind parameter tables,
and utility functions for type conversion and atom property lookup used
throughout the TorchDock scoring pipeline.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import numpy as np

# AutoDock4 atom type
AD_TYPE_C    = 0
AD_TYPE_A    = 1
AD_TYPE_N    = 2
AD_TYPE_O    = 3
AD_TYPE_P    = 4
AD_TYPE_S    = 5
AD_TYPE_H    = 6   # non-polar hydrogen
AD_TYPE_F    = 7
AD_TYPE_I    = 8
AD_TYPE_NA   = 9
AD_TYPE_OA   = 10
AD_TYPE_SA   = 11
AD_TYPE_HD   = 12
AD_TYPE_Mg   = 13
AD_TYPE_Mn   = 14
AD_TYPE_Zn   = 15
AD_TYPE_Ca   = 16
AD_TYPE_Fe   = 17
AD_TYPE_Cl   = 18
AD_TYPE_Br   = 19
AD_TYPE_Si   = 20  # Silicon
AD_TYPE_At   = 21  # Astatine
AD_TYPE_G0   = 22  # closure of cyclic molecules
AD_TYPE_G1   = 23
AD_TYPE_G2   = 24
AD_TYPE_G3   = 25
AD_TYPE_CG0  = 26
AD_TYPE_CG1  = 27
AD_TYPE_CG2  = 28
AD_TYPE_CG3  = 29
AD_TYPE_W    = 30  # hydrated ligand
AD_TYPE_SIZE = 31

# Element type constants (based on SY_TYPE_* but includes H)
EL_TYPE_H    = 0
EL_TYPE_C    = 1
EL_TYPE_N    = 2
EL_TYPE_O    = 3
EL_TYPE_S    = 4
EL_TYPE_P    = 5
EL_TYPE_F    = 6
EL_TYPE_Cl   = 7
EL_TYPE_Br   = 8
EL_TYPE_I    = 9
EL_TYPE_Si   = 10  # Silicon
EL_TYPE_At   = 11  # Astatine
EL_TYPE_Met  = 12  # Metal
EL_TYPE_Dummy= 13  # Dummy atom
EL_TYPE_SIZE = 14

# X-Score type constants
XS_TYPE_C_H   = 0   # hydrophobic carbon
XS_TYPE_C_P   = 1   # polar carbon
XS_TYPE_N_P   = 2   # polar nitrogen
XS_TYPE_N_D   = 3   # nitrogen donor
XS_TYPE_N_A   = 4   # nitrogen acceptor
XS_TYPE_N_DA  = 5   # nitrogen donor and acceptor
XS_TYPE_O_P   = 6   # polar oxygen
XS_TYPE_O_D   = 7   # oxygen donor
XS_TYPE_O_A   = 8   # oxygen acceptor
XS_TYPE_O_DA  = 9   # oxygen donor and acceptor
XS_TYPE_S_P   = 10  # polar sulfur
XS_TYPE_P_P   = 11  # polar phosphorus
XS_TYPE_F_H   = 12  # hydrophobic fluorine
XS_TYPE_Cl_H  = 13  # hydrophobic chlorine
XS_TYPE_Br_H  = 14  # hydrophobic bromine
XS_TYPE_I_H   = 15  # hydrophobic iodine
XS_TYPE_Si    = 16  # silicon
XS_TYPE_At    = 17  # astatine
XS_TYPE_Met_D = 18  # metal donor
XS_TYPE_C_H_CG0 = 19  # hydrophobic carbon (closure CG0)
XS_TYPE_C_P_CG0 = 20  # polar carbon (closure CG0)
XS_TYPE_G0      = 21  # dummy atom G0
XS_TYPE_C_H_CG1 = 22  # hydrophobic carbon (closure CG1)
XS_TYPE_C_P_CG1 = 23  # polar carbon (closure CG1)
XS_TYPE_G1      = 24  # dummy atom G1
XS_TYPE_C_H_CG2 = 25  # hydrophobic carbon (closure CG2)
XS_TYPE_C_P_CG2 = 26  # polar carbon (closure CG2)
XS_TYPE_G2      = 27  # dummy atom G2
XS_TYPE_C_H_CG3 = 28  # hydrophobic carbon (closure CG3)
XS_TYPE_C_P_CG3 = 29  # polar carbon (closure CG3)
XS_TYPE_G3      = 30  # dummy atom G3
XS_TYPE_W       = 31  # hydrated ligand
XS_TYPE_SIZE    = 32

# Atom kind data: (name, radius, depth, hb_depth, hb_radius, solvation, volume, covalent_radius)
ATOM_KIND_DATA = [
    ("C",   2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030, 0.77),  # 0
    ("A",   2.00000, 0.15000, 0.0, 0.0, -0.00052, 33.51030, 0.77),  # 1
    ("N",   1.75000, 0.16000, 0.0, 0.0, -0.00162, 22.44930, 0.75),  # 2
    ("O",   1.60000, 0.20000, 0.0, 0.0, -0.00251, 17.15730, 0.73),  # 3
    ("P",   2.10000, 0.20000, 0.0, 0.0, -0.00110, 38.79240, 1.06),  # 4
    ("S",   2.00000, 0.20000, 0.0, 0.0, -0.00214, 33.51030, 1.02),  # 5
    ("H",   1.00000, 0.02000, 0.0, 0.0,  0.00051,  0.00000, 0.37),  # 6
    ("F",   1.54500, 0.08000, 0.0, 0.0, -0.00110, 15.44800, 0.71),  # 7
    ("I",   2.36000, 0.55000, 0.0, 0.0, -0.00110, 55.05850, 1.33),  # 8
    ("NA",  1.75000, 0.16000,-5.0, 1.9, -0.00162, 22.44930, 0.75),  # 9
    ("OA",  1.60000, 0.20000,-5.0, 1.9, -0.00251, 17.15730, 0.73),  # 10
    ("SA",  2.00000, 0.20000,-1.0, 2.5, -0.00214, 33.51030, 1.02),  # 11
    ("HD",  1.00000, 0.02000, 1.0, 0.0,  0.00051,  0.00000, 0.37),  # 12
    ("Mg",  0.65000, 0.87500, 0.0, 0.0, -0.00110,  1.56000, 1.30),  # 13
    ("Mn",  0.65000, 0.87500, 0.0, 0.0, -0.00110,  2.14000, 1.39),  # 14
    ("Zn",  0.74000, 0.55000, 0.0, 0.0, -0.00110,  1.70000, 1.31),  # 15
    ("Ca",  0.99000, 0.55000, 0.0, 0.0, -0.00110,  2.77000, 1.74),  # 16
    ("Fe",  0.65000, 0.01000, 0.0, 0.0, -0.00110,  1.84000, 1.25),  # 17
    ("Cl",  2.04500, 0.27600, 0.0, 0.0, -0.00110, 35.82350, 0.99),  # 18
    ("Br",  2.16500, 0.38900, 0.0, 0.0, -0.00110, 42.56610, 1.14),  # 19
    ("Si",  2.30000, 0.20000, 0.0, 0.0, -0.00143, 50.96500, 1.11),  # 20
    ("At",  2.40000, 0.55000, 0.0, 0.0, -0.00110, 57.90580, 1.44),  # 21
    ("G0",  0.00000, 0.00000, 0.0, 0.0,  0.00000,  0.00000, 0.77),  # 22
    ("G1",  0.00000, 0.00000, 0.0, 0.0,  0.00000,  0.00000, 0.77),  # 23
    ("G2",  0.00000, 0.00000, 0.0, 0.0,  0.00000,  0.00000, 0.77),  # 24
    ("G3",  0.00000, 0.00000, 0.0, 0.0,  0.00000,  0.00000, 0.77),  # 25
    ("CG0", 2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030, 0.77),  # 26
    ("CG1", 2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030, 0.77),  # 27
    ("CG2", 2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030, 0.77),  # 28
    ("CG3", 2.00000, 0.15000, 0.0, 0.0, -0.00143, 33.51030, 0.77),  # 29
    ("W",   0.00000, 0.00000, 0.0, 0.0,  0.00000,  0.00000, 0.00),  # 30
]

# Precomputed mapping from AD type to element type (for fast lookup)
AD_TYPE_TO_ELEMENT_TYPE_MAP = {
    AD_TYPE_C:    EL_TYPE_C,
    AD_TYPE_A:    EL_TYPE_C,
    AD_TYPE_N:    EL_TYPE_N,
    AD_TYPE_O:    EL_TYPE_O,
    AD_TYPE_P:    EL_TYPE_P,
    AD_TYPE_S:    EL_TYPE_S,
    AD_TYPE_H:    EL_TYPE_H,
    AD_TYPE_F:    EL_TYPE_F,
    AD_TYPE_I:    EL_TYPE_I,
    AD_TYPE_NA:   EL_TYPE_N,
    AD_TYPE_OA:   EL_TYPE_O,
    AD_TYPE_SA:   EL_TYPE_S,
    AD_TYPE_HD:   EL_TYPE_H,
    AD_TYPE_Mg:   EL_TYPE_Met,
    AD_TYPE_Mn:   EL_TYPE_Met,
    AD_TYPE_Zn:   EL_TYPE_Met,
    AD_TYPE_Ca:   EL_TYPE_Met,
    AD_TYPE_Fe:   EL_TYPE_Met,
    AD_TYPE_Cl:   EL_TYPE_Cl,
    AD_TYPE_Br:   EL_TYPE_Br,
    AD_TYPE_Si:   EL_TYPE_Si,
    AD_TYPE_At:   EL_TYPE_At,
    AD_TYPE_CG0:  EL_TYPE_C,
    AD_TYPE_CG1:  EL_TYPE_C,
    AD_TYPE_CG2:  EL_TYPE_C,
    AD_TYPE_CG3:  EL_TYPE_C,
    AD_TYPE_G0:   EL_TYPE_Dummy,
    AD_TYPE_G1:   EL_TYPE_Dummy,
    AD_TYPE_G2:   EL_TYPE_Dummy,
    AD_TYPE_G3:   EL_TYPE_Dummy,
    AD_TYPE_W:    EL_TYPE_Dummy,
    AD_TYPE_SIZE: EL_TYPE_SIZE,
}

# X-Score van der Waals radii
XS_VDW_RADIIS = np.array([
    1.9,  # XS_TYPE_C_H
    1.9,  # XS_TYPE_C_P
    1.8,  # XS_TYPE_N_P
    1.8,  # XS_TYPE_N_D
    1.8,  # XS_TYPE_N_A
    1.8,  # XS_TYPE_N_DA
    1.7,  # XS_TYPE_O_P
    1.7,  # XS_TYPE_O_D
    1.7,  # XS_TYPE_O_A
    1.7,  # XS_TYPE_O_DA
    2.0,  # XS_TYPE_S_P
    2.1,  # XS_TYPE_P_P
    1.5,  # XS_TYPE_F_H
    1.8,  # XS_TYPE_Cl_H
    2.0,  # XS_TYPE_Br_H
    2.2,  # XS_TYPE_I_H
    2.2,  # XS_TYPE_Si
    2.3,  # XS_TYPE_At
    1.2,  # XS_TYPE_Met_D
    1.9,  # XS_TYPE_C_H_CG0
    1.9,  # XS_TYPE_C_P_CG0
    0.0,  # XS_TYPE_G0
    1.9,  # XS_TYPE_C_H_CG1
    1.9,  # XS_TYPE_C_P_CG1
    0.0,  # XS_TYPE_G1
    1.9,  # XS_TYPE_C_H_CG2
    1.9,  # XS_TYPE_C_P_CG2
    0.0,  # XS_TYPE_G2
    1.9,  # XS_TYPE_C_H_CG3
    1.9,  # XS_TYPE_C_P_CG3
    0.0,  # XS_TYPE_G3
    0.0,  # XS_TYPE_W
    0.0,  # XS_TYPE_SIZE
])


# Atom equivalence data: {element: equivalent_element}
ATOM_EQUIVALENCE_DATA = {
    "Se": "S"
}

# non-AD4 metal names set (metals not in AD4 standard types)
NON_AD_METAL_NAMES = {"Cu", "Na", "K", "Hg", "Co", "U", "Cd", "Ni"}

# Non-heteroatom AD types (carbon and hydrogen types)
NON_HETEROATOM_AD_TYPES = {AD_TYPE_A, AD_TYPE_C, AD_TYPE_H, AD_TYPE_HD,
                           AD_TYPE_CG0, AD_TYPE_CG1, AD_TYPE_CG2, AD_TYPE_CG3}

# Precomputed mapping from atom name to AD type index
ATOM_KIND_DATA_NAME_TO_TYPE_INDEX = {atom_data[0]: idx for idx, atom_data in enumerate(ATOM_KIND_DATA)}

# X-Score type constants for acceptors, donors, and hydrophobics
ACCEPTOR_TYPES = {
    XS_TYPE_N_A, XS_TYPE_N_DA,
    XS_TYPE_O_A, XS_TYPE_O_DA
}

DONOR_TYPES = {
    XS_TYPE_N_D, XS_TYPE_N_DA,
    XS_TYPE_O_D, XS_TYPE_O_DA,
    XS_TYPE_Met_D
}

HYDROPHOBIC_TYPES = {
    XS_TYPE_C_H,
    XS_TYPE_C_H_CG0,
    XS_TYPE_C_H_CG1,
    XS_TYPE_C_H_CG2,
    XS_TYPE_C_H_CG3,
    XS_TYPE_F_H,
    XS_TYPE_Cl_H,
    XS_TYPE_Br_H,
    XS_TYPE_I_H
}

def get_atom_kind_parameters(
    ad_types: "np.ndarray | list[int]",
) -> dict[str, np.ndarray]:
    """Get atom kind parameters (radius, depth, hb_depth, etc.) from AD types.

    Args:
        ad_types: np.ndarray or list[int], array of AD type indices for atoms

    Returns:
        dict: Dictionary containing numpy arrays for each parameter:
            - radius: van der Waals radius
            - depth: well depth
            - hb_depth: hydrogen bond depth
            - hb_radius: hydrogen bond radius
            - solvation: solvation parameter
            - volume: atomic volume
            - covalent_radius: covalent radius
    """
    ad_types = np.asarray(ad_types, dtype=int)
    n_atoms = len(ad_types)

    # Initialize output arrays
    radiuses = np.zeros(n_atoms, dtype=float)
    depths = np.zeros(n_atoms, dtype=float)
    hb_depths = np.zeros(n_atoms, dtype=float)
    hb_radiuses = np.zeros(n_atoms, dtype=float)
    solvations = np.zeros(n_atoms, dtype=float)
    volumes = np.zeros(n_atoms, dtype=float)
    covalent_radiuses = np.zeros(n_atoms, dtype=float)

    # Extract parameters from ATOM_KIND_DATA based on AD type
    for i, ad_type in enumerate(ad_types):
        if 0 <= ad_type < len(ATOM_KIND_DATA):
            atom_data = ATOM_KIND_DATA[ad_type]
            # atom_data format: (name, radius, depth, hb_depth, hb_radius, solvation, volume, covalent_radius)
            radiuses[i] = atom_data[1]
            depths[i] = atom_data[2]
            hb_depths[i] = atom_data[3]
            hb_radiuses[i] = atom_data[4]
            solvations[i] = atom_data[5]
            volumes[i] = atom_data[6]
            covalent_radiuses[i] = atom_data[7]
        else:
            # Invalid AD type, set to 0 (will be detected as error by user)
            pass

    return {
        "radius": radiuses,
        "depth": depths,
        "hb_depth": hb_depths,
        "hb_radius": hb_radiuses,
        "solvation": solvations,
        "volume": volumes,
        "covalent_radius": covalent_radiuses
    }


def ad_is_heteroatom(ad_type: int) -> bool:
    """Check if an AD type is a heteroatom.

    Args:
        ad_type: int, AD type index

    Returns:
        bool: True if the AD type is a heteroatom, False otherwise
    """
    return ad_type not in NON_HETEROATOM_AD_TYPES and ad_type < AD_TYPE_SIZE


def bonded_to_heteroatom(
    atom_idx: int,
    ad_types: np.ndarray,
    conn_mat: np.ndarray,
) -> bool:
    """Check if an atom is bonded to a heteroatom.

    Args:
        atom_idx: int, index of the atom to check
        ad_types: np.ndarray, array of AD type indices for all atoms
        conn_mat: np.ndarray, connectivity matrix (non-zero indicates bond)

    Returns:
        bool: True if directly bonded to a heteroatom, False otherwise
    """
    # find all directly bonded atoms (non-zero entries in connectivity matrix)
    bonded_atom_indices = np.where(conn_mat[atom_idx] != 0)[0]

    # check if any bonded atom is a heteroatom
    for bonded_idx in bonded_atom_indices:
        if ad_is_heteroatom(ad_types[bonded_idx]):
            return True

    return False


def bonded_to_HD(
    atom_idx: int,
    ad_types: np.ndarray,
    conn_mat: np.ndarray,
) -> bool:
    """Check if an atom is bonded to a polar hydrogen (HD type).

    Args:
        atom_idx: int, index of the atom to check
        ad_types: np.ndarray, array of AD type indices for all atoms
        conn_mat: np.ndarray, connectivity matrix (non-zero indicates bond)

    Returns:
        bool: True if directly bonded to HD type hydrogen, False otherwise
    """
    # find all directly bonded atoms (non-zero entries in connectivity matrix)
    bonded_atom_indices = np.where(conn_mat[atom_idx] != 0)[0]

    # check if any bonded atom is HD type
    for bonded_idx in bonded_atom_indices:
        if ad_types[bonded_idx] == AD_TYPE_HD:
            return True

    return False


def str_to_ad_type(
    ad_type_strs: list[str],
    ad_types: np.ndarray,
    xs_types: np.ndarray | None = None,
    validate: bool = False,
) -> None:
    """Convert AD type strings to AD type indices.

    Process logic for each atom:
    1. Direct lookup in atom_kind_data_name_to_type_index
    2. If not found, try equivalence mapping in atom_equivalence_data
    3. If it's a non-AD4 metal (Cu/Na/K/Hg/Co/U/Cd/Ni), set xs_type to XS_TYPE_Met_D
    4. If validate=True, check if the type is acceptable (ad_type < AD_TYPE_SIZE or xs_type == Met_D)

    Args:
        ad_type_strs: list[str], AD type strings for each atom
        ad_types: np.ndarray, output array for AD type indices (will be modified in-place)
        xs_types: np.ndarray or None, output array for X-Score type indices (will be modified in-place if provided)
        validate: bool, whether to validate the types (default: False)

    Raises:
        ValueError: if validate=True and an atom type is not acceptable
    """
    for i, name in enumerate(ad_type_strs):
        # direct lookup
        if name in ATOM_KIND_DATA_NAME_TO_TYPE_INDEX:
            ad_type = ATOM_KIND_DATA_NAME_TO_TYPE_INDEX[name]
        else:
            # try equivalence mapping
            if name in ATOM_EQUIVALENCE_DATA:
                equiv_name = ATOM_EQUIVALENCE_DATA[name]
                ad_type = ATOM_KIND_DATA_NAME_TO_TYPE_INDEX.get(equiv_name, AD_TYPE_SIZE)
            else:
                ad_type = AD_TYPE_SIZE

        ad_types[i] = ad_type

        # handle non-AD4 metals
        xs_type = -1
        if name in NON_AD_METAL_NAMES:
            xs_type = XS_TYPE_Met_D
            if xs_types is not None:
                xs_types[i] = xs_type

        # optional validation
        if validate and not (ad_type < AD_TYPE_SIZE or xs_type == XS_TYPE_Met_D):
            raise ValueError(
                f"Unacceptable atom type at index {i}: name='{name}', "
                f"ad_type={ad_type}, xs_type={xs_type}."
            )


def ad_type_to_el_type(
    ad_types: np.ndarray,
    el_types: np.ndarray,
) -> None:
    """Convert AD type indices to element type indices.

    Args:
        ad_types: np.ndarray, input array of AD type indices
        el_types: np.ndarray, output array of element type indices (will be modified in-place)
    """
    for i, ad_type in enumerate(ad_types):
        el_types[i] = AD_TYPE_TO_ELEMENT_TYPE_MAP.get(ad_type, EL_TYPE_SIZE)


def el_type_to_xs_type(
    ad_types: np.ndarray,
    el_types: np.ndarray,
    xs_types: np.ndarray,
    conn_mat: np.ndarray,
) -> None:
    """Convert element type indices to X-Score type indices.

    It assigns XS types based on element type, donor/acceptor properties, and bonding environment.

    Args:
        ad_types: np.ndarray, input array of AD type indices
        el_types: np.ndarray, input array of element type indices
        xs_types: np.ndarray, output array of X-Score type indices (will be modified in-place)
        conn_mat: np.ndarray, connectivity matrix (used to determine bonding)
    """
    for i in range(len(el_types)):
        # skip if xs_type already set (e.g., for non-AD4 metals)
        if xs_types[i] != -1:
            continue

        ad_type = ad_types[i]
        el_type = el_types[i]

        # determine acceptor property (OA or NA in AD typing)
        acceptor = (ad_type == AD_TYPE_OA or ad_type == AD_TYPE_NA)

        # determine donor property (metal or bonded to HD)
        donor_NorO = (el_type == EL_TYPE_Met or bonded_to_HD(i, ad_types, conn_mat))

        # assign XS type based on element type
        if el_type == EL_TYPE_H:
            # hydrogen atoms don't get XS type assigned (stay at -1 or existing value)
            pass

        elif el_type == EL_TYPE_C:
            # carbon: distinguish by closure type and polarity
            is_bonded_to_het = bonded_to_heteroatom(i, ad_types, conn_mat)

            if ad_type == AD_TYPE_CG0:
                xs_types[i] = XS_TYPE_C_P_CG0 if is_bonded_to_het else XS_TYPE_C_H_CG0
            elif ad_type == AD_TYPE_CG1:
                xs_types[i] = XS_TYPE_C_P_CG1 if is_bonded_to_het else XS_TYPE_C_H_CG1
            elif ad_type == AD_TYPE_CG2:
                xs_types[i] = XS_TYPE_C_P_CG2 if is_bonded_to_het else XS_TYPE_C_H_CG2
            elif ad_type == AD_TYPE_CG3:
                xs_types[i] = XS_TYPE_C_P_CG3 if is_bonded_to_het else XS_TYPE_C_H_CG3
            else:
                xs_types[i] = XS_TYPE_C_P if is_bonded_to_het else XS_TYPE_C_H

        elif el_type == EL_TYPE_N:
            # nitrogen: distinguish by donor/acceptor properties
            if acceptor and donor_NorO:
                xs_types[i] = XS_TYPE_N_DA
            elif acceptor:
                xs_types[i] = XS_TYPE_N_A
            elif donor_NorO:
                xs_types[i] = XS_TYPE_N_D
            else:
                xs_types[i] = XS_TYPE_N_P

        elif el_type == EL_TYPE_O:
            # oxygen: distinguish by donor/acceptor properties
            if acceptor and donor_NorO:
                xs_types[i] = XS_TYPE_O_DA
            elif acceptor:
                xs_types[i] = XS_TYPE_O_A
            elif donor_NorO:
                xs_types[i] = XS_TYPE_O_D
            else:
                xs_types[i] = XS_TYPE_O_P

        elif el_type == EL_TYPE_S:
            xs_types[i] = XS_TYPE_S_P

        elif el_type == EL_TYPE_P:
            xs_types[i] = XS_TYPE_P_P

        elif el_type == EL_TYPE_F:
            xs_types[i] = XS_TYPE_F_H

        elif el_type == EL_TYPE_Cl:
            xs_types[i] = XS_TYPE_Cl_H

        elif el_type == EL_TYPE_Br:
            xs_types[i] = XS_TYPE_Br_H

        elif el_type == EL_TYPE_I:
            xs_types[i] = XS_TYPE_I_H

        elif el_type == EL_TYPE_Si:
            xs_types[i] = XS_TYPE_Si

        elif el_type == EL_TYPE_At:
            xs_types[i] = XS_TYPE_At

        elif el_type == EL_TYPE_Met:
            xs_types[i] = XS_TYPE_Met_D

        elif el_type == EL_TYPE_Dummy:
            # dummy atoms: map based on AD type
            if ad_type == AD_TYPE_G0:
                xs_types[i] = XS_TYPE_G0
            elif ad_type == AD_TYPE_G1:
                xs_types[i] = XS_TYPE_G1
            elif ad_type == AD_TYPE_G2:
                xs_types[i] = XS_TYPE_G2
            elif ad_type == AD_TYPE_G3:
                xs_types[i] = XS_TYPE_G3
            elif ad_type == AD_TYPE_W:
                xs_types[i] = XS_TYPE_SIZE  # W has no XS type
            # else: leave as -1 or existing value

        # EL_TYPE_SIZE: no action needed
