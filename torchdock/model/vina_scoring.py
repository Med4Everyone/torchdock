"""
Vina and Vinardo scoring functions for molecular docking.

This module implements the AutoDock Vina and Vinardo scoring functions as
differentiable PyTorch modules. It computes pairwise interaction energies
between ligand and receptor atoms, supporting both rigid and flexible
receptor docking with batched GPU acceleration.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import torch
import torch.nn as nn

from torchdock.constants.vina_constants import ACCEPTOR_TYPES, DONOR_TYPES, HYDROPHOBIC_TYPES
from torchdock.constants.vinardo_constants import (
    VINARDO_GAUSS_OFFSET,
    VINARDO_GAUSS_WIDTH,
    VINARDO_HBOND_BAD,
    VINARDO_HYDROPHOBIC_BAD,
    VINARDO_HYDROPHOBIC_GOOD,
    VINARDO_WEIGHT_GAUSS1,
    VINARDO_WEIGHT_HYDROGEN,
    VINARDO_WEIGHT_HYDROPHOBIC,
    VINARDO_WEIGHT_REPULSION,
    VINARDO_WEIGHT_ROT,
)
from torchdock.data.vina_dataloader import VinaLigandLoader, VinaProteinLoader

class VinaScoreModel(nn.Module):
    """Differentiable Vina/Vinardo scoring model for molecular docking.

    Computes pairwise interaction energies between ligand and receptor atoms
    using either the AutoDock Vina or Vinardo scoring function. Supports
    rigid and flexible receptor docking with batched GPU acceleration,
    distance-based pair filtering, and optional pair caching for performance.
    """

    def __init__(self, config) -> None:
        super(VinaScoreModel, self).__init__()

        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        self.score_function = config.score_function
        
        self.box_center = torch.tensor(config.box_center, dtype=self.dtype, device=self.device)
        self.box_size = torch.tensor(config.box_size, dtype=self.dtype, device=self.device)

        # Initialize score parameters based on score function type
        if self.score_function == 'vinardo':
            # Vinardo parameters
            self.gaussian1_mean = torch.tensor(float(VINARDO_GAUSS_OFFSET), dtype=self.dtype, device=self.device)
            self.gaussian1_sigma = torch.tensor(VINARDO_GAUSS_WIDTH, dtype=self.dtype, device=self.device)
            # Vinardo has only one gaussian term
            self.gaussian2_mean = None
            self.gaussian2_sigma = None
            
            # For Vinardo hydrophobic: slope_step(bad=2.5, good=0, d)
            # Note: VINARDO_HYDROPHOBIC_BAD=0, VINARDO_HYDROPHOBIC_GOOD=2.5 in constants
            # But in slope_step, 'good' is the lower threshold, 'bad' is the upper threshold
            self.hydrophobic_threshold_low = torch.tensor(VINARDO_HYDROPHOBIC_BAD, dtype=self.dtype, device=self.device)  # 0
            self.hydrophobic_threshold_high = torch.tensor(VINARDO_HYDROPHOBIC_GOOD, dtype=self.dtype, device=self.device)  # 2.5
            # For Vinardo H-bond: slope_step(bad=0, good=-0.6, d)
            self.hbond_threshold = torch.tensor(VINARDO_HBOND_BAD, dtype=self.dtype, device=self.device)  # -0.6
            
            # Vinardo weights (no gauss2)
            self.weight_gauss1 = nn.Parameter(torch.tensor(VINARDO_WEIGHT_GAUSS1, dtype=self.dtype, requires_grad=True, device=self.device))
            self.weight_gauss2 = None  # Vinardo has no Gauss2 term
            self.weight_repulsion = nn.Parameter(torch.tensor(VINARDO_WEIGHT_REPULSION, dtype=self.dtype, requires_grad=True, device=self.device))
            self.weight_hydrophobic = nn.Parameter(torch.tensor(VINARDO_WEIGHT_HYDROPHOBIC, dtype=self.dtype, requires_grad=True, device=self.device))
            self.weight_hydrogen = nn.Parameter(torch.tensor(VINARDO_WEIGHT_HYDROGEN, dtype=self.dtype, requires_grad=True, device=self.device))
            self.weight_torsion = nn.Parameter(torch.tensor(VINARDO_WEIGHT_ROT, dtype=self.dtype, requires_grad=True, device=self.device))
        else:
            # Vina parameters (default)
            self.gaussian1_mean = torch.tensor(0., dtype=self.dtype, device=self.device)
            self.gaussian1_sigma = torch.tensor(0.5, dtype=self.dtype, device=self.device)
            self.gaussian2_mean = torch.tensor(3., dtype=self.dtype, device=self.device)
            self.gaussian2_sigma = torch.tensor(2., dtype=self.dtype, device=self.device)
            
            self.hydrophobic_threshold_low = torch.tensor(0.5, dtype=self.dtype, device=self.device)
            self.hydrophobic_threshold_high = torch.tensor(1.5, dtype=self.dtype, device=self.device)
            self.hbond_threshold = torch.tensor(-0.7, dtype=self.dtype, device=self.device)
            
            # Vina weights
            self.weight_gauss1 = nn.Parameter(torch.tensor(-0.035579, dtype=self.dtype, requires_grad=True, device=self.device))
            self.weight_gauss2 = nn.Parameter(torch.tensor(-0.005156, dtype=self.dtype, requires_grad=True, device=self.device))
            self.weight_repulsion = nn.Parameter(torch.tensor(0.840245, dtype=self.dtype, requires_grad=True, device=self.device))
            self.weight_hydrophobic = nn.Parameter(torch.tensor(-0.035069, dtype=self.dtype, requires_grad=True, device=self.device))
            self.weight_hydrogen = nn.Parameter(torch.tensor(-0.587439, dtype=self.dtype, requires_grad=True, device=self.device))
            self.weight_torsion = nn.Parameter(torch.tensor(0.05846, dtype=self.dtype, requires_grad=True, device=self.device))
        
        self.repulsion_pow = torch.tensor(2, dtype=self.dtype, device=self.device)
        self.distance_threshold = torch.tensor(8.0, dtype=self.dtype, device=self.device)
        self.distance_threshold_squared = self.distance_threshold * self.distance_threshold  # Pre-compute squared threshold
        
        # Pocket boundary penalty weight
        self.weight_pocket_penalty = nn.Parameter(torch.tensor(5.0, dtype=self.dtype, requires_grad=False, device=self.device))
        
        # Pair caching mechanism for optimization
        self._cached_pairs = {}  # Store pair indices
        self._static_properties_cache = {}  # Store static properties (VDW radii, H-bond flags, etc.)
        self._use_cached_pairs = False  # Flag to enable/disable cache mode

    def _prepare_ligand_intra_static(self, ligand: VinaLigandLoader) -> None:
        """Precompute static properties for ligand intra-molecular pairs.

        Computes pair-level features that do not depend on coordinates,
        including VDW radii sums, H-bond flags, and hydrophobic flags.

        Args:
            ligand: Loaded ligand molecule containing intra_pairs_index.

        Side effects:
            Sets self.ligand_intra_pairs, self.ligand_intra_vdw_radii_sum,
            self.ligand_intra_is_hbond, and self.ligand_intra_is_hydrophobic.
        """
        if hasattr(ligand, 'intra_pairs_index') and ligand.intra_pairs_index is not None:
            self.ligand_intra_pairs = torch.tensor(ligand.intra_pairs_index, dtype=torch.long, device=self.device)
            n_intra_pairs = self.ligand_intra_pairs.shape[0]
            if n_intra_pairs > 0:
                atom_i = self.ligand_intra_pairs[:, 0]
                atom_j = self.ligand_intra_pairs[:, 1]
                # VDW radii sum
                self.ligand_intra_vdw_radii_sum = self.ligand_xs_vdw_radii[atom_i] + self.ligand_xs_vdw_radii[atom_j]
                # H-bond flags
                is_donor_i = self.ligand_is_donor[atom_i]
                is_donor_j = self.ligand_is_donor[atom_j]
                is_acceptor_i = self.ligand_is_acceptor[atom_i]
                is_acceptor_j = self.ligand_is_acceptor[atom_j]
                self.ligand_intra_is_hbond = (is_donor_i & is_acceptor_j) | (is_acceptor_i & is_donor_j)
                # Hydrophobic flags
                is_hydrophobic_i = self.ligand_is_hydrophobic[atom_i]
                is_hydrophobic_j = self.ligand_is_hydrophobic[atom_j]
                self.ligand_intra_is_hydrophobic = is_hydrophobic_i & is_hydrophobic_j
            else:
                # No intra pairs
                self.ligand_intra_vdw_radii_sum = torch.empty(0, dtype=self.dtype, device=self.device)
                self.ligand_intra_is_hbond = torch.empty(0, dtype=torch.bool, device=self.device)
                self.ligand_intra_is_hydrophobic = torch.empty(0, dtype=torch.bool, device=self.device)
        else:
            # No intra pairs
            self.ligand_intra_pairs = torch.empty((0, 2), dtype=torch.long, device=self.device)
            self.ligand_intra_vdw_radii_sum = torch.empty(0, dtype=self.dtype, device=self.device)
            self.ligand_intra_is_hbond = torch.empty(0, dtype=torch.bool, device=self.device)
            self.ligand_intra_is_hydrophobic = torch.empty(0, dtype=torch.bool, device=self.device)

    def _prepare_flex_rigid_static(self, protein: VinaProteinLoader) -> None:
        """Precompute static properties for flex-rigid inter-molecular pairs.

        Computes pair-level features that do not depend on coordinates,
        including VDW radii sums, H-bond flags, and hydrophobic flags.

        Args:
            protein: Loaded protein containing flex_rigid_pair_indices.

        Side effects:
            Sets self.flex_rigid_pairs, self.flex_rigid_flat_pairs,
            self.flex_rigid_vdw_radii_sum, self.flex_rigid_is_hbond,
            and self.flex_rigid_is_hydrophobic.
        """
        if hasattr(protein, 'flex_rigid_pair_indices') and protein.flex_rigid_pair_indices is not None:
            self.flex_rigid_pairs = torch.tensor(protein.flex_rigid_pair_indices, dtype=torch.long, device=self.device)
            n_flex_rigid_pairs = self.flex_rigid_pairs.shape[0]
            
            if n_flex_rigid_pairs > 0:
                # Extract indices from pairs
                flex_res_idx = self.flex_rigid_pairs[:, 0]    
                flex_local_idx = self.flex_rigid_pairs[:, 1]  
                rigid_idx = self.flex_rigid_pairs[:, 2]       
                
                # Build flattened view for flex–rigid pairs
                flex_flat_idx = flex_res_idx * self.max_m + flex_local_idx 
                self.flex_rigid_flat_pairs = torch.stack((flex_flat_idx, rigid_idx), dim=1)

                # Get flex atom properties
                flex_vdw_radii = self.flex_xs_vdw_radii[flex_res_idx, flex_local_idx]  
                flex_is_donor = self.flex_is_donor[flex_res_idx, flex_local_idx]      
                flex_is_acceptor = self.flex_is_acceptor[flex_res_idx, flex_local_idx]
                flex_is_hydrophobic = self.flex_is_hydrophobic[flex_res_idx, flex_local_idx] 
                
                # Get rigid atom properties 
                rigid_vdw_radii = self.rigid_xs_vdw_radii[rigid_idx]        
                rigid_is_donor = self.rigid_is_donor[rigid_idx]             
                rigid_is_acceptor = self.rigid_is_acceptor[rigid_idx]       
                rigid_is_hydrophobic = self.rigid_is_hydrophobic[rigid_idx] 
                
                # VDW radii sum
                self.flex_rigid_vdw_radii_sum = flex_vdw_radii + rigid_vdw_radii 
                
                # H-bond flags
                self.flex_rigid_is_hbond = (flex_is_donor & rigid_is_acceptor) | (flex_is_acceptor & rigid_is_donor)
                
                # Hydrophobic flags
                self.flex_rigid_is_hydrophobic = flex_is_hydrophobic & rigid_is_hydrophobic
            else:
                # No flex-rigid pairs
                self.flex_rigid_vdw_radii_sum = torch.empty(0, dtype=self.dtype, device=self.device)
                self.flex_rigid_is_hbond = torch.empty(0, dtype=torch.bool, device=self.device)
                self.flex_rigid_is_hydrophobic = torch.empty(0, dtype=torch.bool, device=self.device)
        else:
            # No flex-rigid pairs
            self.flex_rigid_pairs = torch.empty((0, 3), dtype=torch.long, device=self.device)
            self.flex_rigid_vdw_radii_sum = torch.empty(0, dtype=self.dtype, device=self.device)
            self.flex_rigid_is_hbond = torch.empty(0, dtype=torch.bool, device=self.device)
            self.flex_rigid_is_hydrophobic = torch.empty(0, dtype=torch.bool, device=self.device)

    def _prepare_flex_flex_static(self, protein: VinaProteinLoader) -> None:
        """Precompute static properties for flex-flex inter-molecular pairs.

        Computes pair-level features that do not depend on coordinates,
        including VDW radii sums, H-bond flags, and hydrophobic flags.

        Args:
            protein: Loaded protein containing flex_flex_pair_indices.

        Side effects:
            Sets self.flex_flex_pairs, self.flex_flex_flat_pairs,
            self.flex_flex_vdw_radii_sum, self.flex_flex_is_hbond,
            and self.flex_flex_is_hydrophobic.
        """
        if hasattr(protein, 'flex_flex_pair_indices') and protein.flex_flex_pair_indices is not None:
            self.flex_flex_pairs = torch.tensor(protein.flex_flex_pair_indices, dtype=torch.long, device=self.device)
            n_flex_flex_pairs = self.flex_flex_pairs.shape[0]
            
            if n_flex_flex_pairs > 0:
                # Extract indices from pairs
                flex_res_idx_A = self.flex_flex_pairs[:, 0]    
                flex_local_idx_A = self.flex_flex_pairs[:, 1]  
                flex_res_idx_B = self.flex_flex_pairs[:, 2]    
                flex_local_idx_B = self.flex_flex_pairs[:, 3]  
                
                # Build flattened view for flex-flex pairs
                flex_flat_idx_A = flex_res_idx_A * self.max_m + flex_local_idx_A 
                flex_flat_idx_B = flex_res_idx_B * self.max_m + flex_local_idx_B
                self.flex_flex_flat_pairs = torch.stack((flex_flat_idx_A, flex_flat_idx_B), dim=1)

                # Get flex atom A properties
                flex_vdw_radii_A = self.flex_xs_vdw_radii[flex_res_idx_A, flex_local_idx_A]  
                flex_is_donor_A = self.flex_is_donor[flex_res_idx_A, flex_local_idx_A]      
                flex_is_acceptor_A = self.flex_is_acceptor[flex_res_idx_A, flex_local_idx_A]
                flex_is_hydrophobic_A = self.flex_is_hydrophobic[flex_res_idx_A, flex_local_idx_A] 
                
                # Get flex atom B properties 
                flex_vdw_radii_B = self.flex_xs_vdw_radii[flex_res_idx_B, flex_local_idx_B]        
                flex_is_donor_B = self.flex_is_donor[flex_res_idx_B, flex_local_idx_B]             
                flex_is_acceptor_B = self.flex_is_acceptor[flex_res_idx_B, flex_local_idx_B]       
                flex_is_hydrophobic_B = self.flex_is_hydrophobic[flex_res_idx_B, flex_local_idx_B] 
                
                # VDW radii sum
                self.flex_flex_vdw_radii_sum = flex_vdw_radii_A + flex_vdw_radii_B 
                
                # H-bond flags
                self.flex_flex_is_hbond = (flex_is_donor_A & flex_is_acceptor_B) | (flex_is_acceptor_A & flex_is_donor_B)
                
                # Hydrophobic flags
                self.flex_flex_is_hydrophobic = flex_is_hydrophobic_A & flex_is_hydrophobic_B
            else:
                # No flex-flex pairs
                self.flex_flex_vdw_radii_sum = torch.empty(0, dtype=self.dtype, device=self.device)
                self.flex_flex_is_hbond = torch.empty(0, dtype=torch.bool, device=self.device)
                self.flex_flex_is_hydrophobic = torch.empty(0, dtype=torch.bool, device=self.device)
        else:
            # No flex-flex pairs
            self.flex_flex_pairs = torch.empty((0, 4), dtype=torch.long, device=self.device)
            self.flex_flex_vdw_radii_sum = torch.empty(0, dtype=self.dtype, device=self.device)
            self.flex_flex_is_hbond = torch.empty(0, dtype=torch.bool, device=self.device)
            self.flex_flex_is_hydrophobic = torch.empty(0, dtype=torch.bool, device=self.device)

    def prepare_complex(self, ligand: VinaLigandLoader, protein: VinaProteinLoader) -> None:
        """Prepare complex data for Vina scoring.

        Extracts and converts static molecular properties from ligand and protein
        loaders to tensors for efficient scoring calculations. Coordinates are not
        included as they will be dynamically updated during forward passes.

        Properties prepared include ligand atom types, VDW radii, H-bond flags,
        hydrophobicity flags, and valid heavy atom masks for ligand, flexible
        receptor, and rigid receptor atoms, as well as per-pair static attributes
        for ligand-intra and flex-rigid candidate pairs.

        Args:
            ligand: Loaded ligand molecule.
            protein: Loaded protein with pocket partitioning.

        Side effects:
            Sets ligand, flex, and rigid receptor property tensors, and calls
            _prepare_ligand_intra_static, _prepare_flex_rigid_static, and
            _prepare_flex_flex_static to populate per-pair attributes.
        """
        # Initialize type sets as tensors
        acceptor_types_t = torch.tensor(sorted(list(ACCEPTOR_TYPES)), dtype=torch.long, device=self.device)
        donor_types_t = torch.tensor(sorted(list(DONOR_TYPES)), dtype=torch.long, device=self.device)
        hydrophobic_types_t = torch.tensor(sorted(list(HYDROPHOBIC_TYPES)), dtype=torch.long, device=self.device)

        # Prepare ligand properties
        self.ligand_xs_types = torch.tensor(ligand.xs_types, dtype=torch.long, device=self.device)
        self.ligand_xs_vdw_radii = torch.tensor(ligand.xs_vdw_radii, dtype=self.dtype, device=self.device)
        self.ligand_valid_heavy_mask = self.ligand_xs_types >= 0 
        self.ligand_is_acceptor = (self.ligand_xs_types.unsqueeze(-1) == acceptor_types_t).any(dim=-1) & self.ligand_valid_heavy_mask
        self.ligand_is_donor = (self.ligand_xs_types.unsqueeze(-1) == donor_types_t).any(dim=-1) & self.ligand_valid_heavy_mask
        self.ligand_is_hydrophobic = (self.ligand_xs_types.unsqueeze(-1) == hydrophobic_types_t).any(dim=-1) & self.ligand_valid_heavy_mask
                
        # Prepare flexible receptor properties
        if hasattr(protein, 'flex_coords') and protein.flex_coords.size > 0:
            self.flex_xs_types = torch.tensor(protein.flex_xs_types, dtype=torch.long, device=self.device)
            self.flex_xs_vdw_radii = torch.tensor(protein.flex_xs_vdw_radii, dtype=self.dtype, device=self.device)
            self.flex_valid_heavy_mask = self.flex_xs_types >= 0
            self.flex_movable_heavy_mask = torch.tensor(protein.flex_movable_heavy_mask, dtype=torch.bool, device=self.device)
            self.flex_is_acceptor = (self.flex_xs_types.unsqueeze(-1) == acceptor_types_t).any(dim=-1) & self.flex_valid_heavy_mask
            self.flex_is_donor = (self.flex_xs_types.unsqueeze(-1) == donor_types_t).any(dim=-1) & self.flex_valid_heavy_mask
            self.flex_is_hydrophobic = (self.flex_xs_types.unsqueeze(-1) == hydrophobic_types_t).any(dim=-1) & self.flex_valid_heavy_mask
            self.num_flex_residues, max_m = self.flex_xs_types.shape
            self.max_m = max_m
        else:
            # No flexible residues
            self.num_flex_residues = 0
            self.flex_xs_types = torch.empty((0, 0), dtype=torch.long, device=self.device)
            self.flex_xs_vdw_radii = torch.empty((0, 0), dtype=self.dtype, device=self.device)
            self.flex_is_acceptor = torch.empty((0, 0), dtype=torch.bool, device=self.device)
            self.flex_is_donor = torch.empty((0, 0), dtype=torch.bool, device=self.device)
            self.flex_is_hydrophobic = torch.empty((0, 0), dtype=torch.bool, device=self.device)
            self.flex_valid_heavy_mask = torch.empty((0, 0), dtype=torch.bool, device=self.device)
            self.flex_movable_heavy_mask = torch.empty((0, 0), dtype=torch.bool, device=self.device)
        
        # Prepare rigid receptor properties
        if hasattr(protein, 'rigid_coords') and protein.rigid_coords.size > 0:
            self.rigid_xs_types = torch.tensor(protein.rigid_xs_types, dtype=torch.long, device=self.device)
            self.rigid_xs_vdw_radii = torch.tensor(protein.rigid_xs_vdw_radii, dtype=self.dtype, device=self.device)
            self.rigid_valid_heavy_mask = self.rigid_xs_types >= 0
            self.rigid_is_acceptor = (self.rigid_xs_types.unsqueeze(-1) == acceptor_types_t).any(dim=-1) & self.rigid_valid_heavy_mask
            self.rigid_is_donor = (self.rigid_xs_types.unsqueeze(-1) == donor_types_t).any(dim=-1) & self.rigid_valid_heavy_mask
            self.rigid_is_hydrophobic = (self.rigid_xs_types.unsqueeze(-1) == hydrophobic_types_t).any(dim=-1) & self.rigid_valid_heavy_mask
            self._rigid_coords_cached = torch.tensor(protein.rigid_coords, dtype=self.dtype, device=self.device)
        else:
            # No rigid atoms
            self.rigid_xs_types = torch.empty(0, dtype=torch.long, device=self.device)
            self.rigid_xs_vdw_radii = torch.empty(0, dtype=self.dtype, device=self.device)
            self.rigid_is_acceptor = torch.empty(0, dtype=torch.bool, device=self.device)
            self.rigid_is_donor = torch.empty(0, dtype=torch.bool, device=self.device)
            self.rigid_is_hydrophobic = torch.empty(0, dtype=torch.bool, device=self.device)
            self.rigid_valid_heavy_mask = torch.empty(0, dtype=torch.bool, device=self.device)
            self._rigid_coords_cached = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        
        # Per-pair static attributes for the ligand-intra and flex-rigid candidate pairs
        self._prepare_ligand_intra_static(ligand)
        self._prepare_flex_rigid_static(protein)     
        self._prepare_flex_flex_static(protein)   

    def _compute_ligand_intra_pairs(self, ligand_coords: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute ligand intra-molecular pair data for energy calculation.

        Only computes coordinate-dependent distances. Static properties
        (VDW radii sum, H-bond flags, hydrophobic flags) are pre-computed
        in prepare_complex.

        Args:
            ligand_coords: Batch of ligand coordinates, shape (B, N_lig, 3).

        Returns:
            Dictionary with keys 'batch_idx', 'distances', 'vdw_radii_sum',
            'is_hbond', and 'is_hydrophobic', each a tensor of shape (n_valid_pairs,).
        """
        # Fast path: use cached pairs (only recompute distances)
        if self._use_cached_pairs and 'ligand_intra' in self._cached_pairs:
            cache = self._cached_pairs['ligand_intra']
            static = self._static_properties_cache['ligand_intra']
            
            # Extract cached indices
            batch_idx = cache['batch_idx']
            atom_i = cache['atom_i']
            atom_j = cache['atom_j']
            
            # Recompute distances using cached indices
            coords_i = ligand_coords[batch_idx, atom_i]  # [n_cached_pairs, 3]
            coords_j = ligand_coords[batch_idx, atom_j]  # [n_cached_pairs, 3]
            
            diffs = coords_i - coords_j
            distances = torch.sqrt(torch.sum(diffs * diffs, dim=-1) + 1e-10)
            
            # Return cached result with recomputed distances
            # Note: No distance filtering here (based on memory guidance)
            return {
                'batch_idx': batch_idx,
                'distances': distances,
                'vdw_radii_sum': static['vdw_radii_sum'],
                'is_hbond': static['is_hbond'],
                'is_hydrophobic': static['is_hydrophobic']
            }
        
        # Full path: compute all pairs and filter by distance
        B = ligand_coords.shape[0]
        
        if self.ligand_intra_pairs.numel() == 0 :
            # No intra pairs - return empty data
            return {
                'batch_idx': torch.empty(0, dtype=torch.long, device=self.device),
                'distances': torch.empty(0, dtype=self.dtype, device=self.device),
                'vdw_radii_sum': torch.empty(0, dtype=self.dtype, device=self.device),
                'is_hbond': torch.empty(0, dtype=torch.bool, device=self.device),
                'is_hydrophobic': torch.empty(0, dtype=torch.bool, device=self.device)
            }
        
        # Extract atom indices from pairs
        atom_i = self.ligand_intra_pairs[:, 0]
        atom_j = self.ligand_intra_pairs[:, 1]
        
        # Gather coordinates for all batches
        coords_i = ligand_coords[:, atom_i] 
        coords_j = ligand_coords[:, atom_j]
        
        # Compute squared pairwise distances for all batches
        diffs = coords_i - coords_j 
        d2 = torch.sum(diffs * diffs, dim=-1)
        
        # Apply distance threshold mask: keep pairs within cutoff
        valid_mask = d2 <= self.distance_threshold_squared
        
        # Extract valid indices (batch_idx, pair_idx)
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)
        
        if valid_indices.shape[0] == 0:
            # No valid intra pairs within distance threshold
            return {
                'batch_idx': torch.empty(0, dtype=torch.long, device=self.device),
                'distances': torch.empty(0, dtype=self.dtype, device=self.device),
                'vdw_radii_sum': torch.empty(0, dtype=self.dtype, device=self.device),
                'is_hbond': torch.empty(0, dtype=torch.bool, device=self.device),
                'is_hydrophobic': torch.empty(0, dtype=torch.bool, device=self.device)
            }
        
        batch_idx = valid_indices[:, 0] 
        pair_idx = valid_indices[:, 1]
        
        # Gather squared distances using valid indices and take sqrt
        d2_values = d2[batch_idx, pair_idx]
        distances_flat = torch.sqrt(d2_values + 1e-10)
        
        # Pre-computed static properties for all pairs
        vdw_radii_sum_all = self.ligand_intra_vdw_radii_sum
        is_hbond_all = self.ligand_intra_is_hbond
        is_hydrophobic_all = self.ligand_intra_is_hydrophobic
        
        # Filter static properties according to valid pair indices
        vdw_radii_sum = vdw_radii_sum_all[pair_idx]
        is_hbond = is_hbond_all[pair_idx]
        is_hydrophobic = is_hydrophobic_all[pair_idx]
        
        # Cache the results if not in cache mode
        if not self._use_cached_pairs:
            self._cached_pairs['ligand_intra'] = {
                'batch_idx': batch_idx,
                'atom_i': atom_i[pair_idx],
                'atom_j': atom_j[pair_idx]
            }
            self._static_properties_cache['ligand_intra'] = {
                'vdw_radii_sum': vdw_radii_sum,
                'is_hbond': is_hbond,
                'is_hydrophobic': is_hydrophobic
            }
        
        return {
            'batch_idx': batch_idx,
            'distances': distances_flat,
            'vdw_radii_sum': vdw_radii_sum,
            'is_hbond': is_hbond,
            'is_hydrophobic': is_hydrophobic
        }
    
    def _compute_ligand_rigid_inter_pairs(
        self, ligand_coords: torch.Tensor, protein_rigid_coords: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute ligand-rigid receptor inter-molecular pair data for energy calculation.

        Computes distance-filtered pairs between ligand and rigid receptor atoms,
        filtering by distance threshold and heavy atom masks.

        Args:
            ligand_coords: Batch of ligand coordinates including H, shape (B, N_lig, 3).
            protein_rigid_coords: Batch of rigid receptor coordinates including H, shape (B, N_rigid, 3).

        Returns:
            Dictionary with keys 'batch_idx', 'distances', 'vdw_radii_sum',
            'is_hbond', and 'is_hydrophobic', each a tensor of shape (n_valid_pairs,).
        """
        # Fast path: use cached pairs (only recompute distances)
        if self._use_cached_pairs and 'ligand_rigid' in self._cached_pairs:
            cache = self._cached_pairs['ligand_rigid']
            static = self._static_properties_cache['ligand_rigid']
            
            # Extract cached indices
            batch_idx = cache['batch_idx']
            lig_idx = cache['lig_idx']
            rigid_idx = cache['rigid_idx']
            
            # Recompute distances using cached indices
            coords_lig = ligand_coords[batch_idx, lig_idx]  # [n_cached_pairs, 3]
            coords_rigid = protein_rigid_coords[batch_idx, rigid_idx]  # [n_cached_pairs, 3]
            
            diffs = coords_lig - coords_rigid
            distances = torch.sqrt(torch.sum(diffs * diffs, dim=-1) + 1e-10)
            
            # Return cached result with recomputed distances
            # Note: No distance filtering here (based on memory guidance)
            return {
                'batch_idx': batch_idx,
                'distances': distances,
                'vdw_radii_sum': static['vdw_radii_sum'],
                'is_hbond': static['is_hbond'],
                'is_hydrophobic': static['is_hydrophobic'],
            }
        
        # Full path: compute all pairs and filter by distance
        B, N_lig, _ = ligand_coords.shape
        
        if protein_rigid_coords.numel() == 0:
            # No rigid atoms
            return {
                'batch_idx': torch.empty(0, dtype=torch.long, device=self.device),
                'distances': torch.empty(0, dtype=self.dtype, device=self.device),
                'vdw_radii_sum': torch.empty(0, dtype=self.dtype, device=self.device),
                'is_hbond': torch.empty(0, dtype=torch.bool, device=self.device),
                'is_hydrophobic': torch.empty(0, dtype=torch.bool, device=self.device),
            }
        
        N_rigid = protein_rigid_coords.shape[1]
        
        # Compute squared distance matrix
        diffs = ligand_coords.unsqueeze(2) - protein_rigid_coords.unsqueeze(1)
        d2 = torch.sum(diffs * diffs, dim=-1)
        
        # Build valid mask: distance threshold + heavy atom masks
        valid_mask = (d2 <= self.distance_threshold_squared)
        valid_mask = valid_mask & self.ligand_valid_heavy_mask.unsqueeze(0).unsqueeze(2)
        valid_mask = valid_mask & self.rigid_valid_heavy_mask.unsqueeze(0).unsqueeze(1)
        
        # Extract valid pair indices
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)
        
        if valid_indices.shape[0] == 0:
            # No valid pairs found
            return {
                'batch_idx': torch.empty(0, dtype=torch.long, device=self.device),
                'distances': torch.empty(0, dtype=self.dtype, device=self.device),
                'vdw_radii_sum': torch.empty(0, dtype=self.dtype, device=self.device),
                'is_hbond': torch.empty(0, dtype=torch.bool, device=self.device),
                'is_hydrophobic': torch.empty(0, dtype=torch.bool, device=self.device),
            }
        
        batch_idx = valid_indices[:, 0]
        lig_idx = valid_indices[:, 1]
        rigid_idx = valid_indices[:, 2]
        
        # Gather distances from d2 matrix and compute sqrt
        d2_values = d2[batch_idx, lig_idx, rigid_idx]
        distances = torch.sqrt(d2_values + 1e-10)
        
        # Gather VDW radii sum using atom indices
        vdw_radii_sum = self.ligand_xs_vdw_radii[lig_idx] + self.rigid_xs_vdw_radii[rigid_idx]
        
        # Gather H-bond flags
        is_hbond = (
            (self.ligand_is_donor[lig_idx] & self.rigid_is_acceptor[rigid_idx]) |
            (self.ligand_is_acceptor[lig_idx] & self.rigid_is_donor[rigid_idx])
        )
        
        # Gather hydrophobic flags
        is_hydrophobic = self.ligand_is_hydrophobic[lig_idx] & self.rigid_is_hydrophobic[rigid_idx]
        
        # Cache the results if not in cache mode
        if not self._use_cached_pairs:
            self._cached_pairs['ligand_rigid'] = {
                'batch_idx': batch_idx,
                'lig_idx': lig_idx,
                'rigid_idx': rigid_idx
            }
            self._static_properties_cache['ligand_rigid'] = {
                'vdw_radii_sum': vdw_radii_sum,
                'is_hbond': is_hbond,
                'is_hydrophobic': is_hydrophobic
            }
        
        return {
            'batch_idx': batch_idx,
            'distances': distances,
            'vdw_radii_sum': vdw_radii_sum,
            'is_hbond': is_hbond,
            'is_hydrophobic': is_hydrophobic,
        }
    
    def _compute_ligand_flex_inter_pairs(
        self, ligand_coords: torch.Tensor, protein_flex_coords: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute ligand-flexible receptor inter-molecular pair data for energy calculation.

        Computes distance-filtered pairs between ligand and flexible receptor atoms.
        Flexible residues are flattened for efficient distance computation.

        Args:
            ligand_coords: Batch of ligand coordinates including H, shape (B, N_lig, 3).
            protein_flex_coords: Batch of flexible residue coordinates including H and padding,
                shape (B, F, max_m, 3).

        Returns:
            Dictionary with keys 'batch_idx', 'distances', 'vdw_radii_sum',
            'is_hbond', and 'is_hydrophobic', each a tensor of shape (n_valid_pairs,).
        """
        # Fast path: use cached pairs (only recompute distances)
        if self._use_cached_pairs and 'ligand_flex' in self._cached_pairs:
            cache = self._cached_pairs['ligand_flex']
            static = self._static_properties_cache['ligand_flex']
            
            # Extract cached indices
            batch_idx = cache['batch_idx']
            lig_idx = cache['lig_idx']
            flex_idx = cache['flex_idx']
            
            # Flatten flexible residue coordinates
            F, max_m = protein_flex_coords.shape[1], protein_flex_coords.shape[2]
            protein_flex_coords_flat = protein_flex_coords.reshape(protein_flex_coords.shape[0], F * max_m, 3)
            
            # Recompute distances using cached indices
            coords_lig = ligand_coords[batch_idx, lig_idx]  # [n_cached_pairs, 3]
            coords_flex = protein_flex_coords_flat[batch_idx, flex_idx]  # [n_cached_pairs, 3]
            
            diffs = coords_lig - coords_flex
            distances = torch.sqrt(torch.sum(diffs * diffs, dim=-1) + 1e-10)
            
            # Return cached result with recomputed distances
            # Note: No distance filtering here (based on memory guidance)
            return {
                'batch_idx': batch_idx,
                'distances': distances,
                'vdw_radii_sum': static['vdw_radii_sum'],
                'is_hbond': static['is_hbond'],
                'is_hydrophobic': static['is_hydrophobic'],
            }
        
        # Full path: compute all pairs and filter by distance
        B, N_lig, _ = ligand_coords.shape
        
        # Check if there are flexible residues
        if protein_flex_coords.numel() == 0:
            # No flexible residues
            return {
                'batch_idx': torch.empty(0, dtype=torch.long, device=self.device),
                'distances': torch.empty(0, dtype=self.dtype, device=self.device),
                'vdw_radii_sum': torch.empty(0, dtype=self.dtype, device=self.device),
                'is_hbond': torch.empty(0, dtype=torch.bool, device=self.device),
                'is_hydrophobic': torch.empty(0, dtype=torch.bool, device=self.device),
            }
        
        # Flatten flexible residue coordinates
        F, max_m = protein_flex_coords.shape[1], protein_flex_coords.shape[2]
        protein_flex_coords_flat = protein_flex_coords.reshape(B, F * max_m, 3)
        N_flex = F * max_m
        
        # Flatten flexible properties
        flex_xs_vdw_radii_flat = self.flex_xs_vdw_radii.reshape(-1)
        flex_valid_heavy_mask_flat = self.flex_valid_heavy_mask.reshape(-1)
        flex_is_acceptor_flat = self.flex_is_acceptor.reshape(-1)
        flex_is_donor_flat = self.flex_is_donor.reshape(-1)
        flex_is_hydrophobic_flat = self.flex_is_hydrophobic.reshape(-1)
        
        # Compute squared distance matrix
        diffs = ligand_coords.unsqueeze(2) - protein_flex_coords_flat.unsqueeze(1)
        d2 = torch.sum(diffs * diffs, dim=-1)
        
        # Build valid mask: distance threshold + heavy atom masks
        valid_mask = (d2 <= self.distance_threshold_squared)
        valid_mask = valid_mask & self.ligand_valid_heavy_mask.unsqueeze(0).unsqueeze(2)
        valid_mask = valid_mask & flex_valid_heavy_mask_flat.unsqueeze(0).unsqueeze(1)
        
        # Extract valid pair indices
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)
        
        if valid_indices.shape[0] == 0:
            # No valid pairs found
            return {
                'batch_idx': torch.empty(0, dtype=torch.long, device=self.device),
                'distances': torch.empty(0, dtype=self.dtype, device=self.device),
                'vdw_radii_sum': torch.empty(0, dtype=self.dtype, device=self.device),
                'is_hbond': torch.empty(0, dtype=torch.bool, device=self.device),
                'is_hydrophobic': torch.empty(0, dtype=torch.bool, device=self.device),
            }
        
        batch_idx = valid_indices[:, 0]
        lig_idx = valid_indices[:, 1]
        flex_idx = valid_indices[:, 2]
        
        # Gather distances from d2 matrix and compute sqrt
        d2_values = d2[batch_idx, lig_idx, flex_idx]
        distances = torch.sqrt(d2_values + 1e-10)
        
        # Gather VDW radii sum using atom indices (static properties, no batch dimension)
        vdw_radii_sum = self.ligand_xs_vdw_radii[lig_idx] + flex_xs_vdw_radii_flat[flex_idx]
        
        # Gather H-bond flags
        is_hbond = (
            (self.ligand_is_donor[lig_idx] & flex_is_acceptor_flat[flex_idx]) |
            (self.ligand_is_acceptor[lig_idx] & flex_is_donor_flat[flex_idx])
        )
        
        # Gather hydrophobic flags
        is_hydrophobic = self.ligand_is_hydrophobic[lig_idx] & flex_is_hydrophobic_flat[flex_idx]
        
        # Cache the results if not in cache mode
        if not self._use_cached_pairs:
            self._cached_pairs['ligand_flex'] = {
                'batch_idx': batch_idx,
                'lig_idx': lig_idx,
                'flex_idx': flex_idx
            }
            self._static_properties_cache['ligand_flex'] = {
                'vdw_radii_sum': vdw_radii_sum,
                'is_hbond': is_hbond,
                'is_hydrophobic': is_hydrophobic
            }
        
        return {
            'batch_idx': batch_idx,
            'distances': distances,
            'vdw_radii_sum': vdw_radii_sum,
            'is_hbond': is_hbond,
            'is_hydrophobic': is_hydrophobic,
        }
    
    def _compute_flex_rigid_inter_pairs(
        self, protein_flex_coords: torch.Tensor, protein_rigid_coords: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute flexible-rigid receptor inter-molecular pair data for energy calculation.

        Uses precomputed static candidate pairs and applies the distance threshold
        at runtime to filter pairs.

        Args:
            protein_flex_coords: Batch of flexible residue coordinates including H and padding,
                shape (B, F, max_m, 3).
            protein_rigid_coords: Batch of rigid receptor coordinates including H, shape (B, N_rigid, 3).

        Returns:
            Dictionary with keys 'batch_idx', 'distances', 'vdw_radii_sum',
            'is_hbond', and 'is_hydrophobic', each a tensor of shape (n_valid_pairs,).
        """
        # Fast path: use cached pairs (only recompute distances)
        if self._use_cached_pairs and 'flex_rigid' in self._cached_pairs:
            cache = self._cached_pairs['flex_rigid']
            static = self._static_properties_cache['flex_rigid']
            
            # Extract cached indices
            batch_idx = cache['batch_idx']
            pair_idx = cache['pair_idx']
            
            # Flatten flexible residue coordinates
            F, max_m = protein_flex_coords.shape[1], protein_flex_coords.shape[2]
            protein_flex_coords_flat = protein_flex_coords.reshape(protein_flex_coords.shape[0], F * max_m, 3)
            
            # Extract flattened indices from precomputed candidate pairs
            flex_flat_idx = self.flex_rigid_flat_pairs[pair_idx, 0]
            rigid_idx_static = self.flex_rigid_flat_pairs[pair_idx, 1]
            
            # Recompute distances using cached indices
            coords_flex = protein_flex_coords_flat[batch_idx, flex_flat_idx]  # [n_cached_pairs, 3]
            coords_rigid = protein_rigid_coords[batch_idx, rigid_idx_static]  # [n_cached_pairs, 3]
            
            diffs = coords_flex - coords_rigid
            distances = torch.sqrt(torch.sum(diffs * diffs, dim=-1) + 1e-10)
            
            # Return cached result with recomputed distances
            # Note: No distance filtering here (based on memory guidance)
            return {
                'batch_idx': batch_idx,
                'distances': distances,
                'vdw_radii_sum': static['vdw_radii_sum'],
                'is_hbond': static['is_hbond'],
                'is_hydrophobic': static['is_hydrophobic'],
            }
        
        # Full path: compute candidate pairs and filter by distance
        B = protein_rigid_coords.shape[0]
        
        # Early return if no candidate pairs
        if self.flex_rigid_pairs.numel() == 0:
            # No flex-rigid pairs - return empty data
            return {
                'batch_idx': torch.empty(0, dtype=torch.long, device=self.device),
                'distances': torch.empty(0, dtype=self.dtype, device=self.device),
                'vdw_radii_sum': torch.empty(0, dtype=self.dtype, device=self.device),
                'is_hbond': torch.empty(0, dtype=torch.bool, device=self.device),
                'is_hydrophobic': torch.empty(0, dtype=torch.bool, device=self.device),
            }
        
        # Flatten flexible residue coordinates
        F, max_m = protein_flex_coords.shape[1], protein_flex_coords.shape[2]
        protein_flex_coords_flat = protein_flex_coords.reshape(B, F * max_m, 3)
        
        # Extract flattened indices from precomputed candidate pairs
        flex_flat_idx = self.flex_rigid_flat_pairs[:, 0]
        rigid_idx_static = self.flex_rigid_flat_pairs[:, 1]
        
        # Gather coordinates for candidate pairs across batches
        coords_flex = protein_flex_coords_flat[:, flex_flat_idx]
        coords_rigid = protein_rigid_coords[:, rigid_idx_static]
        
        # Compute squared distances for candidate pairs
        d2 = torch.sum((coords_flex - coords_rigid) ** 2, dim=-1)
        
        # Apply distance threshold
        valid_mask = d2 <= self.distance_threshold_squared
        
        # Extract valid indices
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)
        
        if valid_indices.shape[0] == 0:
            # No valid pairs within distance threshold
            return {
                'batch_idx': torch.empty(0, dtype=torch.long, device=self.device),
                'distances': torch.empty(0, dtype=self.dtype, device=self.device),
                'vdw_radii_sum': torch.empty(0, dtype=self.dtype, device=self.device),
                'is_hbond': torch.empty(0, dtype=torch.bool, device=self.device),
                'is_hydrophobic': torch.empty(0, dtype=torch.bool, device=self.device),
            }
        
        batch_idx = valid_indices[:, 0]
        pair_idx = valid_indices[:, 1]
        
        # Gather distances for valid pairs and compute sqrt
        d2_values = d2[batch_idx, pair_idx]
        distances = torch.sqrt(d2_values + 1e-10)
        
        # Pre-computed static properties for all candidate pairs
        vdw_radii_sum_all = self.flex_rigid_vdw_radii_sum
        is_hbond_all = self.flex_rigid_is_hbond
        is_hydrophobic_all = self.flex_rigid_is_hydrophobic
        
        # Filter static properties according to valid pair indices
        vdw_radii_sum = vdw_radii_sum_all[pair_idx]
        is_hbond = is_hbond_all[pair_idx]
        is_hydrophobic = is_hydrophobic_all[pair_idx]
        
        # Cache the results if not in cache mode
        if not self._use_cached_pairs:
            self._cached_pairs['flex_rigid'] = {
                'batch_idx': batch_idx,
                'pair_idx': pair_idx
            }
            self._static_properties_cache['flex_rigid'] = {
                'vdw_radii_sum': vdw_radii_sum,
                'is_hbond': is_hbond,
                'is_hydrophobic': is_hydrophobic
            }
        
        return {
            'batch_idx': batch_idx,
            'distances': distances,
            'vdw_radii_sum': vdw_radii_sum,
            'is_hbond': is_hbond,
            'is_hydrophobic': is_hydrophobic,
        }
    
    def _compute_flex_flex_inter_pairs(
        self, protein_flex_coords: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute flexible-flexible inter-residue pair data for energy calculation.

        Uses precomputed static candidate pairs and applies the distance threshold
        at runtime to filter pairs.

        Args:
            protein_flex_coords: Batch of flexible residue coordinates including H and padding,
                shape (B, F, max_m, 3).

        Returns:
            Dictionary with keys 'batch_idx', 'distances', 'vdw_radii_sum',
            'is_hbond', and 'is_hydrophobic', each a tensor of shape (n_valid_pairs,).
        """
        # Fast path: use cached pairs (only recompute distances)
        if self._use_cached_pairs and 'flex_flex' in self._cached_pairs:
            cache = self._cached_pairs['flex_flex']
            static = self._static_properties_cache['flex_flex']
            
            # Extract cached indices
            batch_idx = cache['batch_idx']
            pair_idx = cache['pair_idx']
            
            # Flatten flexible residue coordinates
            F, max_m = protein_flex_coords.shape[1], protein_flex_coords.shape[2]
            protein_flex_coords_flat = protein_flex_coords.reshape(protein_flex_coords.shape[0], F * max_m, 3)
            
            # Extract flattened indices from precomputed candidate pairs
            flex_flat_idx_A = self.flex_flex_flat_pairs[pair_idx, 0]
            flex_flat_idx_B = self.flex_flex_flat_pairs[pair_idx, 1]
            
            # Recompute distances using cached indices
            coords_A = protein_flex_coords_flat[batch_idx, flex_flat_idx_A]  # [n_cached_pairs, 3]
            coords_B = protein_flex_coords_flat[batch_idx, flex_flat_idx_B]  # [n_cached_pairs, 3]
            
            diffs = coords_A - coords_B
            distances = torch.sqrt(torch.sum(diffs * diffs, dim=-1) + 1e-10)
            
            # Return cached result with recomputed distances
            # Note: No distance filtering here (based on memory guidance)
            return {
                'batch_idx': batch_idx,
                'distances': distances,
                'vdw_radii_sum': static['vdw_radii_sum'],
                'is_hbond': static['is_hbond'],
                'is_hydrophobic': static['is_hydrophobic'],
            }
        
        # Full path: compute candidate pairs and filter by distance
        B = protein_flex_coords.shape[0]
        
        # Early return if no candidate pairs
        if self.flex_flex_pairs.numel() == 0:
            # No flex-flex pairs - return empty data
            return {
                'batch_idx': torch.empty(0, dtype=torch.long, device=self.device),
                'distances': torch.empty(0, dtype=self.dtype, device=self.device),
                'vdw_radii_sum': torch.empty(0, dtype=self.dtype, device=self.device),
                'is_hbond': torch.empty(0, dtype=torch.bool, device=self.device),
                'is_hydrophobic': torch.empty(0, dtype=torch.bool, device=self.device),
            }
        
        # Flatten flexible residue coordinates
        F, max_m = protein_flex_coords.shape[1], protein_flex_coords.shape[2]
        protein_flex_coords_flat = protein_flex_coords.reshape(B, F * max_m, 3)
        
        # Extract flattened indices from precomputed candidate pairs
        flex_flat_idx_A = self.flex_flex_flat_pairs[:, 0]
        flex_flat_idx_B = self.flex_flex_flat_pairs[:, 1]
        
        # Gather coordinates for candidate pairs across batches
        coords_A = protein_flex_coords_flat[:, flex_flat_idx_A]
        coords_B = protein_flex_coords_flat[:, flex_flat_idx_B]
        
        # Compute squared distances for candidate pairs
        d2 = torch.sum((coords_A - coords_B) ** 2, dim=-1)
        
        # Apply distance threshold
        valid_mask = d2 <= self.distance_threshold_squared
        
        # Extract valid indices
        valid_indices = torch.nonzero(valid_mask, as_tuple=False)
        
        if valid_indices.shape[0] == 0:
            # No valid pairs within distance threshold
            return {
                'batch_idx': torch.empty(0, dtype=torch.long, device=self.device),
                'distances': torch.empty(0, dtype=self.dtype, device=self.device),
                'vdw_radii_sum': torch.empty(0, dtype=self.dtype, device=self.device),
                'is_hbond': torch.empty(0, dtype=torch.bool, device=self.device),
                'is_hydrophobic': torch.empty(0, dtype=torch.bool, device=self.device),
            }
        
        batch_idx = valid_indices[:, 0]
        pair_idx = valid_indices[:, 1]
        
        # Gather distances for valid pairs and compute sqrt
        d2_values = d2[batch_idx, pair_idx]
        distances = torch.sqrt(d2_values + 1e-10)
        
        # Pre-computed static properties for all candidate pairs
        vdw_radii_sum_all = self.flex_flex_vdw_radii_sum
        is_hbond_all = self.flex_flex_is_hbond
        is_hydrophobic_all = self.flex_flex_is_hydrophobic
        
        # Filter static properties according to valid pair indices
        vdw_radii_sum = vdw_radii_sum_all[pair_idx]
        is_hbond = is_hbond_all[pair_idx]
        is_hydrophobic = is_hydrophobic_all[pair_idx]
        
        # Cache the results if not in cache mode
        if not self._use_cached_pairs:
            self._cached_pairs['flex_flex'] = {
                'batch_idx': batch_idx,
                'pair_idx': pair_idx
            }
            self._static_properties_cache['flex_flex'] = {
                'vdw_radii_sum': vdw_radii_sum,
                'is_hbond': is_hbond,
                'is_hydrophobic': is_hydrophobic
            }
        
        return {
            'batch_idx': batch_idx,
            'distances': distances,
            'vdw_radii_sum': vdw_radii_sum,
            'is_hbond': is_hbond,
            'is_hydrophobic': is_hydrophobic,
        }
    
    def _slope_step(self, x: torch.Tensor, bad: float, good: float) -> torch.Tensor:
        """Compute slope_step function for smooth interpolation.

        Implements a piecewise linear interpolation where values at or below
        'good' return 1, values at or above 'bad' return 0, and intermediate
        values are linearly interpolated.

        Args:
            x: Input values.
            bad: Upper threshold (unfavorable).
            good: Lower threshold (favorable).

        Returns:
            Slope step values in [0, 1].
        """
        return torch.clamp((bad - x) / (bad - good), min=0.0, max=1.0)
    
    def _compute_pair_energy(
        self,
        distances: torch.Tensor,
        vdw_radii_sum: torch.Tensor,
        is_hbond: torch.Tensor,
        is_hydrophobic: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy for a set of atom pairs.

        Supports both Vina (5 energy terms) and Vinardo (4 energy terms)
        scoring functions.

        Args:
            distances: Pairwise distances, shape (n_pairs,).
            vdw_radii_sum: Sum of VDW radii (optimal distance), shape (n_pairs,).
            is_hbond: H-bond interaction flag, shape (n_pairs,).
            is_hydrophobic: Hydrophobic interaction flag, shape (n_pairs,).

        Returns:
            Total energy for each pair, shape (n_pairs,).
        """
        # Early return for empty input
        if distances.numel() == 0:
            return torch.empty(0, dtype=self.dtype, device=self.device)
        
        # Compute d = r - optimal_distance (where optimal_distance = vdw_radii_sum)
        d_radii = distances - vdw_radii_sum
        
        # 1. Gaussian1: exp(-((d - offset) / width)^2)
        gauss1 = torch.exp(
            -torch.pow((d_radii - self.gaussian1_mean) / self.gaussian1_sigma, 2)
        )
        
        if self.score_function == 'vinardo':
            # Vinardo scoring function
            
            # 2. Repulsion: max(0, -d)^2  (equivalent to: d^2 if d < 0, else 0)
            # This is the same as Vina's repulsion formula
            repulsion = torch.where(
                d_radii >= 0,
                torch.tensor(0.0, dtype=self.dtype, device=self.device),
                torch.pow(d_radii, self.repulsion_pow)
            )
            
            # 3. Hydrophobic: slope_step(bad=2.5, good=0, d)
            # When d >= 2.5 -> 0 (too far)
            # When d <= 0 -> 1 (optimal contact)
            hydrophobic = self._slope_step(
                d_radii,
                bad=self.hydrophobic_threshold_high.item(),  # 2.5
                good=self.hydrophobic_threshold_low.item()   # 0
            ) * is_hydrophobic
            
            # 4. H-bond: slope_step(bad=0, good=-0.6, d)
            # When d >= 0 -> 0
            # When d <= -0.6 -> 1
            hbond = self._slope_step(
                d_radii,
                bad=0.0,
                good=self.hbond_threshold.item()  # -0.6
            ) * is_hbond
            
            # Compute weighted sum (Vinardo has no Gauss2 term)
            energies = (
                self.weight_gauss1 * gauss1 +
                self.weight_repulsion * repulsion +
                self.weight_hydrophobic * hydrophobic +
                self.weight_hydrogen * hbond
            )
        else:
            # Vina scoring function (default)
            
            # 2. Gaussian2: exp(-((d - 3) / 2.0)^2)
            gauss2 = torch.exp(
                -torch.pow((d_radii - self.gaussian2_mean) / self.gaussian2_sigma, 2)
            )
            
            # 3. Repulsion: d^2 if d < 0, else 0
            repulsion = torch.where(
                d_radii >= 0,
                torch.tensor(0.0, dtype=self.dtype, device=self.device),
                torch.pow(d_radii, self.repulsion_pow)
            )
            
            # 4. Hydrophobic: slope_step(bad=1.5, good=0.5, d)
            hydrophobic = self._slope_step(
                d_radii,
                bad=self.hydrophobic_threshold_high.item(),  # 1.5
                good=self.hydrophobic_threshold_low.item()   # 0.5
            ) * is_hydrophobic
            
            # 5. H-bond: slope_step(bad=0, good=-0.7, d)
            hbond = self._slope_step(
                d_radii,
                bad=0.0,
                good=self.hbond_threshold.item()  # -0.7
            ) * is_hbond
            
            # Compute weighted sum of all energy terms
            energies = (
                self.weight_gauss1 * gauss1 +
                self.weight_gauss2 * gauss2 +
                self.weight_repulsion * repulsion +
                self.weight_hydrophobic * hydrophobic +
                self.weight_hydrogen * hbond
            )
        
        return energies
    
    def _compute_pocket_boundary_penalty(self, ligand_coords: torch.Tensor) -> torch.Tensor:
        """Compute penalty for ligand atoms outside the pocket boundary.

        The penalty is calculated as the sum of squared distances that heavy atoms
        extend beyond the pocket boundaries.

        Args:
            ligand_coords: Batch of ligand coordinates, shape (B, N_lig, 3).

        Returns:
            Penalty for each sample in batch, shape (B,).
        """
        B = ligand_coords.shape[0]
        
        # Get pocket boundaries (precompute and cache as tensors)
        if not hasattr(self, '_pocket_min'):
            box_center = self.box_center
            box_size = self.box_size
            self._pocket_min = box_center - box_size / 2.0
            self._pocket_max = box_center + box_size / 2.0
        
        # Only penalize heavy atoms
        heavy_mask = self.ligand_valid_heavy_mask
        
        # Gather heavy atom coordinates:
        heavy_coords = ligand_coords[:, heavy_mask, :]
        
        if heavy_coords.shape[1] == 0:
            # No heavy atoms (should not happen in practice)
            return torch.zeros(B, dtype=self.dtype, device=self.device)
        
        # Calculate violations beyond boundary
        penalty_min = torch.clamp(self._pocket_min - heavy_coords, min=0.0)
        penalty_max = torch.clamp(heavy_coords - self._pocket_max, min=0.0)
        
        # Sum penalties across atoms and coordinates: [B]
        pocket_penalty = (penalty_min**2 + penalty_max**2).sum(dim=(1, 2))
        pocket_penalty = self.weight_pocket_penalty * pocket_penalty
        
        return pocket_penalty
    
    def rebuild_pairs_cache(
        self, ligand_coords: torch.Tensor, protein_flex_coords: torch.Tensor
    ) -> None:
        """Rebuild pair cache by computing full pair lists at current coordinates.

        Should be called periodically (e.g., every update_interval steps) to
        refresh the pair cache for subsequent fast forward passes.

        Args:
            ligand_coords: Batch of ligand coordinates, shape (B, N_lig, 3).
            protein_flex_coords: Batch of flexible residue coordinates, shape (B, F, max_m, 3).
        """
        # Temporarily disable cache mode to force full computation
        self._use_cached_pairs = False
        
        # Clear existing cache
        self._cached_pairs = {}
        self._static_properties_cache = {}
        
        # Compute rigid coords (broadcast cached rigid to batch)
        B = ligand_coords.shape[0]
        base_rigid = self._rigid_coords_cached
        if base_rigid is None or base_rigid.numel() == 0:
            protein_rigid_coords = torch.empty((B, 0, 3), dtype=self.dtype, device=self.device)
        else:
            protein_rigid_coords = base_rigid.unsqueeze(0).expand(B, -1, -1)
        
        # Compute all pair types (will populate cache as side effect)
        # Order: from high value to low value
        _ = self._compute_ligand_rigid_inter_pairs(ligand_coords, protein_rigid_coords)
        _ = self._compute_ligand_flex_inter_pairs(ligand_coords, protein_flex_coords)
        _ = self._compute_ligand_intra_pairs(ligand_coords)
        _ = self._compute_flex_rigid_inter_pairs(protein_flex_coords, protein_rigid_coords)
        _ = self._compute_flex_flex_inter_pairs(protein_flex_coords)
        
        # Enable cache mode for subsequent forward passes
        self._use_cached_pairs = True
    
    def clear_pairs_cache(self) -> None:
        """Clear pair cache and disable cache mode.

        Call this when starting a new search attempt or when cache becomes invalid.
        """
        self._cached_pairs = {}
        self._static_properties_cache = {}
        self._use_cached_pairs = False
    
    def forward(
        self,
        ligand_coords: torch.Tensor,
        protein_flex_coords: torch.Tensor,
        include_pocket_penalty: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Vina score for batch inputs using gather-compute-scatter pattern.

        The total score is composed of four main components:
        1. Ligand intra energy: internal conformational strain of the ligand.
        2. Ligand inter energy: ligand-protein interactions (ligand-rigid + ligand-flex).
        3. Protein intra energy: internal strain of flexible residues
           (flex-rigid + flex-flex inter-residue interactions).
        4. Pocket boundary penalty: penalty for ligand atoms extending beyond pocket
           boundaries (only when include_pocket_penalty=True).

        Args:
            ligand_coords: Batch of ligand coordinates including H, shape (B, N_lig, 3).
            protein_flex_coords: Batch of flexible residue coordinates including H and padding,
                shape (B, F, max_m, 3).
            include_pocket_penalty: Whether to add pocket boundary penalty. Defaults to False.

        Returns:
            Tuple of (total_scores, ligand_inter_scores, ligand_intra_scores, protein_intra_scores),
            each a tensor of shape (B,).

        Raises:
            ValueError: If ligand and flex batch sizes do not match.
        """
        B = ligand_coords.shape[0]
        
        # Require K consistency with ligand
        if protein_flex_coords.shape[0] != B:
            raise ValueError(f"Ligand batch size {B} must equal Flex batch size {protein_flex_coords.shape[0]}")

        # Rigid coords: broadcast cached rigid to B
        base_rigid = self._rigid_coords_cached
        if base_rigid is None or base_rigid.numel() == 0:
            protein_rigid_coords = torch.empty((B, 0, 3), dtype=self.dtype, device=self.device)
        else:
            protein_rigid_coords = base_rigid.unsqueeze(0).expand(B, -1, -1)
        
        # Initialize energy components for all batches
        ligand_intra_scores = torch.zeros(B, dtype=self.dtype, device=self.device)
        ligand_inter_scores = torch.zeros(B, dtype=self.dtype, device=self.device)
        protein_intra_scores = torch.zeros(B, dtype=self.dtype, device=self.device)
        pocket_penalty = torch.zeros(B, dtype=self.dtype, device=self.device)
        total_scores = torch.zeros(B, dtype=self.dtype, device=self.device)
        
        # ========== 1. Ligand Intra Energy ==========
        # Compute ligand internal conformational strain
        ligand_intra_data = self._compute_ligand_intra_pairs(ligand_coords)
        
        ligand_intra_energy = self._compute_pair_energy(
            ligand_intra_data['distances'],
            ligand_intra_data['vdw_radii_sum'],
            ligand_intra_data['is_hbond'],
            ligand_intra_data['is_hydrophobic']
        )

        # Aggregate ligand intra energy by batch
        if ligand_intra_energy.numel() > 0:
            ligand_intra_scores.scatter_add_(0, ligand_intra_data['batch_idx'], ligand_intra_energy)

        # ========== 2. Ligand Inter Energy ==========
        # Compute ligand-rigid receptor interactions
        ligand_rigid_data = self._compute_ligand_rigid_inter_pairs(ligand_coords, protein_rigid_coords)
        
        ligand_rigid_energy = self._compute_pair_energy(
            ligand_rigid_data['distances'],
            ligand_rigid_data['vdw_radii_sum'],
            ligand_rigid_data['is_hbond'],
            ligand_rigid_data['is_hydrophobic']
        )
        
        # Aggregate ligand-rigid inter energy
        if ligand_rigid_energy.numel() > 0:
            ligand_inter_scores.scatter_add_(0, ligand_rigid_data['batch_idx'], ligand_rigid_energy)
        
        # Compute ligand-flex receptor interactions
        ligand_flex_data = self._compute_ligand_flex_inter_pairs(ligand_coords, protein_flex_coords)
        
        ligand_flex_energy = self._compute_pair_energy(
            ligand_flex_data['distances'],
            ligand_flex_data['vdw_radii_sum'],
            ligand_flex_data['is_hbond'],
            ligand_flex_data['is_hydrophobic']
        )
        
        # Aggregate ligand-flex inter energy
        if ligand_flex_energy.numel() > 0:
            ligand_inter_scores.scatter_add_(0, ligand_flex_data['batch_idx'], ligand_flex_energy)
        
        # ========== 3. Protein Intra Energy ==========
        # Compute flex-rigid receptor interactions
        flex_rigid_data = self._compute_flex_rigid_inter_pairs(protein_flex_coords, protein_rigid_coords)
        
        flex_rigid_energy = self._compute_pair_energy(
            flex_rigid_data['distances'],
            flex_rigid_data['vdw_radii_sum'],
            flex_rigid_data['is_hbond'],
            flex_rigid_data['is_hydrophobic']
        )
        
        # Aggregate flex-rigid inter energy
        if flex_rigid_energy.numel() > 0:
            protein_intra_scores.scatter_add_(0, flex_rigid_data['batch_idx'], flex_rigid_energy)
        
        # Compute flex-flex receptor interactions
        flex_flex_data = self._compute_flex_flex_inter_pairs(protein_flex_coords)
        
        flex_flex_energy = self._compute_pair_energy(
            flex_flex_data['distances'],
            flex_flex_data['vdw_radii_sum'],
            flex_flex_data['is_hbond'],
            flex_flex_data['is_hydrophobic']
        )
        
        # Aggregate flex-flex inter energy
        if flex_flex_energy.numel() > 0:
            protein_intra_scores.scatter_add_(0, flex_flex_data['batch_idx'], flex_flex_energy)
        
        # Normalize protein intra energy by number of flexible residues
        # If no flexible residues, protein_intra_scores remains 0
        # if self.num_flex_residues > 0:
        #     protein_intra_scores = protein_intra_scores / self.num_flex_residues
        
        # ========== 4. Pocket Boundary Penalty ==========
        if include_pocket_penalty:
            pocket_penalty = self._compute_pocket_boundary_penalty(ligand_coords)
        else:
            pocket_penalty = torch.zeros(B, dtype=self.dtype, device=self.device)
        
        # ========== Total Score ==========
        total_scores = ligand_intra_scores + ligand_inter_scores + protein_intra_scores + pocket_penalty
        
        return total_scores, ligand_inter_scores, ligand_intra_scores, protein_intra_scores

