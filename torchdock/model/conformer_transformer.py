"""
Conformer transformer modules for ligand and protein coordinate generation.

This module implements differentiable coordinate transforms for molecular
docking. LigandConformerTransform applies Position-Orientation-Torsion (POT)
parameterization to generate ligand conformations, while
ProteinConformerTransform applies side-chain chi-angle rotations to flexible
receptor residues. Both modules support batched GPU acceleration and
selective active-sample updates via masking.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import torch
import torch.nn as nn


class TransformModule(nn.Module):
    """Base class for differentiable coordinate transform modules.

    Provides shared device, dtype, and config attributes used by
    ligand and protein conformer transform subclasses.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.dtype = getattr(torch, config.dtype)
        self.device = torch.device(config.device)


class LigandConformerTransform(TransformModule):
    """Ligand conformer transform with Position-Orientation-Torsion (POT) parameterization.

    Generates ligand conformations by applying learnable torsion rotations,
    global orientation quaternion, and translation to a reference conformation.
    Supports batched coordinate transforms with selective active-sample updates.
    """

    def __init__(self, config, ligand_loader) -> None:
        super().__init__(config)
        self.model_name = "LigandConformerTransform"
        self.batch_size = self.config.batch_size
        self.max_torsions = self.config.ligand_max_torsions

        reference_coords = torch.tensor(ligand_loader.ligand_coords, dtype=self.dtype, device=self.device)
        atom_center_coords = torch.tensor(ligand_loader.atom_center_coords, dtype=self.dtype, device=self.device)
        position_center_coords = torch.tensor(ligand_loader.position_center_coords, dtype=self.dtype, device=self.device)
        torsions = torch.tensor(ligand_loader.torsions, dtype=torch.long, device=self.device)
        torsion_masks = torch.tensor(ligand_loader.torsion_masks, dtype=torch.bool, device=self.device)

        self.torsion_count = min(self.max_torsions, len(torsions))
        self.torsion_bond_indices = torsions[:self.torsion_count].tolist()  # [T, 2]

        # Register buffers (non-learnable)
        self.register_buffer("reference_coords", reference_coords)  # [N,3]
        self.register_buffer("atom_center_coords", atom_center_coords)  # [3]
        self.register_buffer("position_center_coords", position_center_coords)  # [3]
        self.register_buffer("torsion_masks", torsion_masks[:self.torsion_count]) # [T, N]

        # POT parameters
        self.position_delta = nn.Parameter(
            torch.zeros(self.batch_size, 3, dtype=self.dtype, device=self.device), 
            requires_grad=True
        )
        self.orientation_quaternion = nn.Parameter(
            torch.zeros(self.batch_size, 4, dtype=self.dtype, device=self.device), 
            requires_grad=True
        )
        with torch.no_grad():
            self.orientation_quaternion[:, 0] = 1.0  # unit quaternion [w, x, y, z] = [1, 0, 0, 0]
        self.torsion_angles = nn.Parameter(
            torch.zeros(self.batch_size, self.torsion_count, dtype=self.dtype, device=self.device), 
            requires_grad=True
        )

    def reset_parameters(self, initial_values: dict[str, torch.Tensor] | None = None) -> None:
        """Reset or set the learnable POT parameters.

        Args:
            initial_values: Optional dict containing initial values for POT parameters.
                Expected keys (all optional): 'position_delta' (B, 3),
                'orientation_quaternion' (B, 4), 'torsion_angles' (B, T).
                Defaults to None.

        Raises:
            TypeError: If initial_values is not a dict or a value is not a Tensor.
            ValueError: If a tensor shape does not match the expected shape.
        """
        with torch.no_grad():
            if initial_values is None:
                # Default initialization
                self.position_delta.zero_()
                self.orientation_quaternion.zero_()
                self.orientation_quaternion[:, 0] = 1.0  # unit quaternion
                self.torsion_angles.zero_()
            else:
                if not isinstance(initial_values, dict):
                    raise TypeError(f"initial_values must be a dict or None, got {type(initial_values)}")

                # Position delta
                if 'position_delta' in initial_values:
                    pd = initial_values['position_delta']
                    if not isinstance(pd, torch.Tensor):
                        raise TypeError("position_delta must be a torch.Tensor")
                    if pd.shape != self.position_delta.shape:
                        raise ValueError(
                            f"position_delta shape mismatch: expected {self.position_delta.shape}, got {pd.shape}"
                        )
                    self.position_delta.copy_(pd.to(dtype=self.dtype, device=self.position_delta.device))

                # Orientation quaternion
                if 'orientation_quaternion' in initial_values:
                    oq = initial_values['orientation_quaternion']
                    if not isinstance(oq, torch.Tensor):
                        raise TypeError("orientation_quaternion must be a torch.Tensor")
                    if oq.shape != self.orientation_quaternion.shape:
                        raise ValueError(
                            f"orientation_quaternion shape mismatch: expected {self.orientation_quaternion.shape}, got {oq.shape}"
                        )
                    # Normalize to unit quaternion for numerical stability
                    norm = torch.norm(oq, dim=-1, keepdim=True).clamp(min=1e-8)
                    oq = oq / norm
                    self.orientation_quaternion.copy_(oq.to(dtype=self.dtype, device=self.orientation_quaternion.device))

                # Torsion angles
                if 'torsion_angles' in initial_values:
                    ta = initial_values['torsion_angles']
                    if not isinstance(ta, torch.Tensor):
                        raise TypeError("torsion_angles must be a torch.Tensor")
                    expected_shape = (self.batch_size, self.torsion_count)
                    if ta.shape != expected_shape:
                        raise ValueError(
                            f"torsion_angles shape mismatch: expected {expected_shape}, got {ta.shape}"
                        )
                    self.torsion_angles.copy_(ta.to(dtype=self.dtype, device=self.torsion_angles.device))

    @staticmethod
    def rotate_by_torsion(
        coords: torch.Tensor,
        angles: torch.Tensor,
        i: int,
        j: int,
        rotation_mask: torch.Tensor,
        reference_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a single torsion rotation using fixed axis from reference.

        Uses Rodrigues' rotation formula to rotate atoms around the bond
        axis defined by reference atoms i and j.

        Args:
            coords: Batch of atomic coordinates, shape (B, N, 3).
            angles: Torsion angles in radians, shape (B,).
            i: First bond atom index (from reference).
            j: Second bond atom index (from reference).
            rotation_mask: Boolean mask of atoms to rotate, shape (N,).
            reference_coords: Reference atomic coordinates, shape (N, 3).

        Returns:
            Rotated coordinates with masked atoms unchanged, shape (B, N, 3).
        """
        B, N = coords.shape[:2]

        p_i = reference_coords[i]  # [3]
        p_j = reference_coords[j]  # [3]

        k = p_j - p_i
        k_norm = torch.norm(k, dim=-1, keepdim=True).clamp(min=1e-8)
        k = k / k_norm  # [3]
        k = k.unsqueeze(0).expand(B, -1)  # [B, 3]

        origin = p_i.unsqueeze(0).expand(B, -1)  # [B, 3]
        v = coords - origin.unsqueeze(1)         # [B, N, 3]

        cos_theta = torch.cos(angles).view(B, 1, 1)
        sin_theta = torch.sin(angles).view(B, 1, 1)
        one_minus_cos = 1.0 - cos_theta

        k_exp = k.unsqueeze(1)  # [B, 1, 3]

        k_dot_v = (k_exp * v).sum(dim=-1, keepdim=True)     # [B, N, 1]
        k_cross_v = torch.cross(k_exp, v, dim=-1)           # [B, N, 3]

        v_rot = (
            v * cos_theta +
            k_cross_v * sin_theta +
            k_exp * k_dot_v * one_minus_cos
        )  # [B, N, 3]

        rotated = v_rot + origin.unsqueeze(1)

        mask_exp = rotation_mask.view(1, N, 1)  # [1, N, 1]
        return torch.where(mask_exp, rotated, coords)

    @staticmethod
    def rotate_by_orientation(
        coords: torch.Tensor,
        center: torch.Tensor,
        quaternion: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate coordinates using pure quaternion math (no rotation matrix).

        Applies quaternion-based rotation around a specified center point
        using the standard q x v x q* formula via two cross products.

        Args:
            coords: Batch of atomic coordinates, shape (B, N, 3).
            center: Rotation center, shape (3,).
            quaternion: Rotation quaternions [w, x, y, z], shape (B, 4).

        Returns:
            Rotated coordinates, shape (B, N, 3).
        """
        quat_norm = torch.norm(quaternion, dim=-1, keepdim=True).clamp(min=1e-8)
        q = quaternion / quat_norm  # [B, 4]
        w = q[:, :1]      # [B, 1]
        vec = q[:, 1:]    # [B, 3]

        v = coords - center.unsqueeze(0)  # [B, N, 3]

        q_exp = vec.unsqueeze(1)   # [B, 1, 3]
        w_exp = w.unsqueeze(1)     # [B, 1, 1]

        # Two cross products (standard and efficient)
        q_cross_v = torch.cross(q_exp, v, dim=-1)               # [B, N, 3]
        q_cross_qcrossv = torch.cross(q_exp, q_cross_v, dim=-1)  # [B, N, 3]

        v_rot = v + 2.0 * w_exp * q_cross_v + 2.0 * q_cross_qcrossv
        return v_rot + center.unsqueeze(0)

    def forward(self, active_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Transform reference ligand via POT for active samples.

        Applies torsion rotations (leaf-to-root), global orientation, and
        translation to generate transformed ligand coordinates.

        Args:
            active_mask: Boolean mask of active batch elements, shape (B,).
                If None, uses all batch elements. Defaults to None.

        Returns:
            Transformed coordinates, shape (K, N, 3) where K is the number
            of True entries in active_mask.
        """
        B = self.position_delta.size(0)  # total batch size

        if active_mask is None:
            active_mask = torch.ones(B, dtype=torch.bool, device=self.position_delta.device)

        K = active_mask.sum().item()

        # Gather active parameters
        pos_delta = self.position_delta[active_mask]          # [K, 3]
        quat = self.orientation_quaternion[active_mask]       # [K, 4]
        torsion_angles = self.torsion_angles[active_mask]     # [K, T]

        coords = self.reference_coords.unsqueeze(0).expand(K, -1, -1)  # [K, N, 3]

        # Apply torsions in reverse order (leaf → root)
        for idx in reversed(range(self.torsion_count)):
            i, j = self.torsion_bond_indices[idx]
            mask = self.torsion_masks[idx]                 # [N]
            angles = torsion_angles[:, idx]                # [K]

            coords = self.rotate_by_torsion(
                coords=coords,
                angles=angles,
                i=i,
                j=j,
                rotation_mask=mask,
                reference_coords=self.reference_coords
            )

        # Apply global orientation
        coords = self.rotate_by_orientation(
            coords=coords,
            center=self.atom_center_coords,  # [3]
            quaternion=quat
        )

        # Apply translation
        coords = coords + pos_delta.unsqueeze(1)  # [K, 1, 3]

        return coords


class ProteinConformerTransform(TransformModule):
    """Protein side-chain conformer transform with chi-angle parameterization.

    Generates flexible residue conformations by applying learnable chi-angle
    rotations (chi1-chi4) to side-chain atoms. Uses vectorized Rodrigues'
    rotation for efficient batched computation across residues and samples.
    """

    def __init__(self, config, protein_loader) -> None:
        super().__init__(config)
        self.model_name = "ProteinConformerTransform"
        self.batch_size = self.config.batch_size

        # Load flexible residue data (padded to max M)
        flex_coords = torch.tensor(protein_loader.flex_coords, dtype=self.dtype, device=self.device)          # [F, M, 3]
        flex_torsions = torch.tensor(protein_loader.flex_torsions, dtype=torch.long, device=self.device)      # [F, 4, 2]
        flex_torsion_masks = torch.tensor(protein_loader.flex_torsion_masks, dtype=torch.bool, device=self.device)  # [F, 4, M]

        self.F, self.M = flex_coords.shape[0], flex_coords.shape[1]

        # Register buffers
        self.register_buffer("flex_coords", flex_coords)                # [F, M, 3]
        self.register_buffer("flex_torsions", flex_torsions)            # [F, 4, 2]
        self.register_buffer("flex_torsion_masks", flex_torsion_masks)  # [F, 4, M]

        # Validity: [F, 4], True if torsion is real (i != -1)
        torsion_valid = (flex_torsions[:, :, 0] != -1)
        self.register_buffer("torsion_valid", torsion_valid)

        # Learnable torsion angles: [batch, F, 4]
        self.protein_torsion_angles = nn.Parameter(
            torch.zeros(self.batch_size, self.F, 4, dtype=self.dtype, device=self.device), 
            requires_grad=True
        )

        # Validate invalid torsions
        self._validate_invalid_torsions()

    def _validate_invalid_torsions(self) -> None:
        """Ensure that for any invalid torsion (i=-1), the rotation mask is all False.

        Raises:
            ValueError: If an invalid torsion has True entries in its rotation mask.
        """
        invalid_torsion_mask = (self.flex_torsions[:, :, 0] == -1)
        
        if invalid_torsion_mask.any():
            # Gather masks for all invalid torsions: [num_invalid, M]
            invalid_masks = self.flex_torsion_masks[invalid_torsion_mask]
            # Check if any of them has a True entry
            if invalid_masks.any():
                # Find first offending (f, chi) for error message
                f_idx, chi_idx = torch.where(invalid_torsion_mask)
                for i in range(len(f_idx)):
                    f, chi = f_idx[i].item(), chi_idx[i].item()
                    if self.flex_torsion_masks[f, chi].any():
                        raise ValueError(
                            f"Invalid torsion at residue {f}, chi {chi}: "
                            f"bond indices = {self.flex_torsions[f, chi].tolist()}, "
                            f"but rotation mask has {self.flex_torsion_masks[f, chi].sum().item()} True entries. "
                            "Expected all False for invalid torsions."
                        )

    def reset_parameters(self, initial_values: dict[str, torch.Tensor] | None = None) -> None:
        """Reset or set the learnable protein side-chain torsion parameters.

        Args:
            initial_values: Optional dict containing initial values.
                Expected key (optional): 'protein_torsion_angles' (B, F, 4) in radians.
                Defaults to None.

        Raises:
            TypeError: If initial_values is not a dict or the value is not a Tensor.
            ValueError: If the tensor shape does not match the expected shape.
        """
        with torch.no_grad():
            if initial_values is None:
                # Default: reset to reference conformation (all chi angles = 0)
                self.protein_torsion_angles.zero_()
            else:
                if not isinstance(initial_values, dict):
                    raise TypeError(f"initial_values must be a dict or None, got {type(initial_values)}")

                # Protein torsion angles
                if 'protein_torsion_angles' in initial_values:
                    ta = initial_values['protein_torsion_angles']
                    if not isinstance(ta, torch.Tensor):
                        raise TypeError("protein_torsion_angles must be a torch.Tensor")
                    
                    expected_shape = (self.batch_size, self.F, 4)
                    if ta.shape != expected_shape:
                        raise ValueError(
                            f"protein_torsion_angles shape mismatch: "
                            f"expected {expected_shape}, got {ta.shape}"
                        )
                    
                    self.protein_torsion_angles.copy_(ta.to(dtype=self.dtype, device=self.protein_torsion_angles.device))

    @staticmethod
    def rotate_protein_chi_vectorized(
        coords: torch.Tensor,
        angles: torch.Tensor,
        bond_indices: torch.Tensor,
        rotation_mask: torch.Tensor,
        reference_coords: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply vectorized chi-angle rotation for all valid residues at once.

        Uses Rodrigues' rotation formula on valid residues only, skipping
        invalid entries for efficiency.

        Args:
            coords: Batch of residue coordinates, shape (K, F, M, 3).
            angles: Torsion angles in radians, shape (K, F).
            bond_indices: Bond atom indices (i, j) per residue, shape (F, 2).
                Invalid entries have i=-1.
            rotation_mask: Boolean mask of atoms to rotate per residue, shape (F, M).
            reference_coords: Reference residue coordinates, shape (F, M, 3).
            valid_mask: Boolean mask of residues with valid torsion at this chi level,
                shape (F,).

        Returns:
            Rotated coordinates, shape (K, F, M, 3).
        """
        K, F, M = coords.shape[:3]
        device = coords.device

        # Only process valid residues
        if not valid_mask.any():
            return coords

        # Extract valid residue indices
        valid_idx = torch.nonzero(valid_mask, as_tuple=True)[0]  # [Fv]
        Fv = valid_idx.size(0)

        # Slice valid part (key: angles_v contains only valid angles!)
        coords_v = coords[:, valid_idx]                          # [K, Fv, M, 3]
        angles_v = angles[:, valid_idx]                          # [K, Fv]
        bond_v = bond_indices[valid_idx]                         # [Fv, 2]
        mask_v = rotation_mask[valid_idx]                        # [Fv, M]
        ref_v = reference_coords[valid_idx]                      # [Fv, M, 3]

        # Safe indexing (now bond_v has i,j >= 0)
        arange = torch.arange(Fv, device=coords.device)
        i_idx = bond_v[:, 0]
        j_idx = bond_v[:, 1]
        p_i = ref_v[arange, i_idx]
        p_j = ref_v[arange, j_idx]

        # Normal Rodrigues rotation (only on valid entries)
        k = (p_j - p_i)
        k = k / torch.norm(k, dim=-1, keepdim=True).clamp(min=1e-8)
        
        k_exp = k.unsqueeze(0).expand(K, -1, -1)
        origin = p_i.unsqueeze(0).expand(K, -1, -1)
        v = coords_v - origin.unsqueeze(2)

        ang = angles_v.unsqueeze(-1).unsqueeze(-1)
        cos_a = torch.cos(ang)
        sin_a = torch.sin(ang)
        omc = 1.0 - cos_a

        k3 = k_exp.unsqueeze(2)
        k_dot_v = (k3 * v).sum(dim=-1, keepdim=True)
        k_cross_v = torch.cross(k3.expand_as(v), v, dim=-1)

        v_rot = v * cos_a + k_cross_v * sin_a + k3 * k_dot_v * omc
        rotated_v = v_rot + origin.unsqueeze(2)

        # Apply rotation mask
        out_v = torch.where(mask_v.unsqueeze(0).unsqueeze(-1), rotated_v, coords_v)

        # Scatter back to full shape
        output = coords.clone()
        output[:, valid_idx] = out_v
        return output
        
    def forward(self, active_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply side-chain chi-angle rotations for active samples.

        Applies rotations in reverse chi order (chi4 -> chi3 -> chi2 -> chi1)
        using vectorized Rodrigues' formula on valid residues.

        Args:
            active_mask: Boolean mask of active batch elements, shape (B,).
                If None, uses all batch elements. Defaults to None.

        Returns:
            Transformed flexible residue coordinates, shape (K, F, M, 3)
            where K is the number of True entries in active_mask.
        """
        B = self.protein_torsion_angles.size(0)  # total batch size

        if active_mask is None:
            active_mask = torch.ones(B, dtype=torch.bool, device=self.protein_torsion_angles.device)

        K = active_mask.sum().item()
        angles = self.protein_torsion_angles[active_mask]  # [K, F, 4]
        coords = self.flex_coords.unsqueeze(0).expand(K, -1, -1, -1)  # [K, F, M, 3]

        # Apply χ4 → χ3 → χ2 → χ1
        for chi in reversed(range(4)):
            bond_indices = self.flex_torsions[:, chi]        # [F, 2]
            masks = self.flex_torsion_masks[:, chi]          # [F, M]
            valid = self.torsion_valid[:, chi]               # [F]
            current_angles = angles[:, :, chi]               # [K, F]

            coords = self.rotate_protein_chi_vectorized(
                coords=coords,
                angles=current_angles,
                bond_indices=bond_indices,
                rotation_mask=masks,
                reference_coords=self.flex_coords,
                valid_mask=valid
            )

        return coords