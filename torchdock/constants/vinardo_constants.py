"""
Vinardo scoring function weights, potential parameters, and VdW radii.

Defines default weights and geometric parameters for the five Vinardo
potential terms (Gaussian, repulsion, hydrophobic, hydrogen bond, and
rotational penalty), along with X-Score type van der Waals radii used
by the Vinardo scoring implementation.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import numpy as np

# ============================================================================
# Vinardo Scoring Function Weights (Default)
# ============================================================================
VINARDO_WEIGHT_GAUSS1 = -0.045
VINARDO_WEIGHT_REPULSION = 0.8
VINARDO_WEIGHT_HYDROPHOBIC = -0.035
VINARDO_WEIGHT_HYDROGEN = -0.600
VINARDO_WEIGHT_GLUE = 50.0
VINARDO_WEIGHT_ROT = 0.05846

# ============================================================================
# Vinardo Potential Function Parameters
# ============================================================================
# 1. Gaussian Potential
# vinardo_gaussian(offset=0, width=0.8, cutoff=8.0)
VINARDO_GAUSS_OFFSET = 0       # Added to optimal distance (Angstrom)
VINARDO_GAUSS_WIDTH = 0.8       # Gaussian width parameter (Angstrom)
VINARDO_GAUSS_CUTOFF = 8.0      # Cutoff distance (Angstrom)

# 2. Repulsion Potential
# vinardo_repulsion(offset=0, cutoff=8.0)
VINARDO_REPULSION_OFFSET = 0.0  # Added to vdw radius sum (Angstrom)
VINARDO_REPULSION_CUTOFF = 8.0  # Cutoff distance (Angstrom)

# 3. Hydrophobic Potential
# vinardo_hydrophobic(good=0, bad=2.5, cutoff=8.0)
VINARDO_HYDROPHOBIC_BAD = 0     # Lower threshold for slope_step (Angstrom)
VINARDO_HYDROPHOBIC_GOOD = 2.5  # Upper threshold for slope_step (Angstrom)
VINARDO_HYDROPHOBIC_CUTOFF = 8.0  # Cutoff distance (Angstrom)

# 4. Hydrogen Bond Potential
# vinardo_non_dir_h_bond(good=-0.6, bad=0, cutoff=8.0)
VINARDO_HBOND_BAD = -0.6        # Lower threshold for slope_step (Angstrom)
VINARDO_HBOND_GOOD = 0          # Upper threshold for slope_step (Angstrom)
VINARDO_HBOND_CUTOFF = 8.0      # Cutoff distance (Angstrom)

# General cutoff settings
VINARDO_CUTOFF = 8.0            # Main cutoff (Angstrom)

# Vinardo XS Type VdW Radii
VINARDO_XS_VDW_RADII = np.array([
    2.0,  # 0:  XS_TYPE_C_H       
    2.0,  # 1:  XS_TYPE_C_P       
    1.7,  # 2:  XS_TYPE_N_P       
    1.7,  # 3:  XS_TYPE_N_D       
    1.7,  # 4:  XS_TYPE_N_A       
    1.7,  # 5:  XS_TYPE_N_DA      
    1.6,  # 6:  XS_TYPE_O_P       
    1.6,  # 7:  XS_TYPE_O_D       
    1.6,  # 8:  XS_TYPE_O_A       
    1.6,  # 9:  XS_TYPE_O_DA      
    2.0,  # 10: XS_TYPE_S_P       
    2.1,  # 11: XS_TYPE_P_P       
    1.5,  # 12: XS_TYPE_F_H       
    1.8,  # 13: XS_TYPE_Cl_H      
    2.0,  # 14: XS_TYPE_Br_H      
    2.2,  # 15: XS_TYPE_I_H       
    2.2,  # 16: XS_TYPE_Si        
    2.3,  # 17: XS_TYPE_At        
    1.2,  # 18: XS_TYPE_Met_D     
    2.0,  # 19: XS_TYPE_C_H_CG0   
    2.0,  # 20: XS_TYPE_C_P_CG0   
    0.0,  # 21: XS_TYPE_G0  
    2.0,  # 22: XS_TYPE_C_H_CG1   
    2.0,  # 23: XS_TYPE_C_P_CG1   
    0.0,  # 24: XS_TYPE_G1   
    2.0,  # 25: XS_TYPE_C_H_CG2   
    2.0,  # 26: XS_TYPE_C_P_CG2   
    0.0,  # 27: XS_TYPE_G2   
    2.0,  # 28: XS_TYPE_C_H_CG3   
    2.0,  # 29: XS_TYPE_C_P_CG3   
    0.0,  # 30: XS_TYPE_G3        
    0.0,  # 31: XS_TYPE_W 
    0.0,  # 32: XS_TYPE_SIZE        
])