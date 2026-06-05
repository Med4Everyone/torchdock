"""
Docking result post-processing and output formatting.

This module provides abstract and concrete processors for saving docking
results (ligand/protein conformations, scores, RMSDs) in various output
formats, including the AutoDock Vina PDBQT format.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import os
import shutil
import uuid
from abc import ABC, abstractmethod
from typing import Any

import torch

from torchdock.metrics.rmsd import calculate_rmsd


class DockingResultProcessor(ABC):
    """Abstract base class for docking result post-processing.

    Handles saving ligand/protein conformations in different formats.
    Subclasses must implement :meth:`process_and_save_results`.
    """

    def __init__(
        self,
        config: Any,
        ligand_loader: Any,
        protein_loader: Any,
        scoring_model: Any,
    ) -> None:
        """Initialize the result processor.

        Args:
            config: Configuration object with docking parameters.
            ligand_loader: Ligand data loader instance.
            protein_loader: Protein data loader instance.
            scoring_model: Scoring model instance.
        """
        self.config = config
        self.ligand_loader = ligand_loader
        self.protein_loader = protein_loader
        self.scoring_model = scoring_model
        self.max_torsions: int = config.ligand_max_torsions
        self.logger = config.logger
        self.score_function: str = config.score_function

        # Check if this is flexible docking.
        self.is_flexible_docking: bool = protein_loader.flex_coords.shape[0] > 0

        # Get weight_torsion and degrees of freedom for score calculation.
        self.weight_torsion: float = scoring_model.weight_torsion.item()
        self.degrees_of_freedom: int = ligand_loader.degrees_of_freedom

    @abstractmethod
    def process_and_save_results(
        self, selected_poses: list[dict], output_path: str
    ) -> list[float]:
        """Process selected poses and save to specified format.

        Args:
            selected_poses: Pose dictionaries from final evaluation.
            output_path: Path to output file.

        Returns:
            List of scores: ``[torchdock_score, total_score, inter_score,
            intra_score, unbound_score]``.
        """
        raise NotImplementedError("Subclass must implement process_and_save_results method.")

    def _calculate_unbound_energy(self, selected_poses: list[dict]) -> float:
        """Calculate UNBOUND energy (INTRA of the best pose, i.e., rank 1).

        Args:
            selected_poses: List of pose dictionaries, sorted by total_score.

        Returns:
            UNBOUND energy value.
        """
        best_pose = selected_poses[0]
        unbound = best_pose['ligand_intra_score'] + best_pose['protein_intra_score']
        return float(unbound)

    def _save_ligand_pdbqt(
        self, ligand_coords: torch.Tensor, output_path: str
    ) -> None:
        """Save ligand PDBQT file with updated coordinates.

        Args:
            ligand_coords: Ligand coordinates of shape ``[N_lig, 3]``.
            output_path: Path to save PDBQT file.
        """
        if torch.is_tensor(ligand_coords):
            ligand_coords = ligand_coords.cpu().numpy()
        self.ligand_loader.save_pdbqt(output_path, coordinates=ligand_coords)

    def _save_pocket_pdbqt(
        self, protein_coords: torch.Tensor, output_path: str
    ) -> None:
        """Save flexible pocket PDBQT file with updated coordinates.

        Args:
            protein_coords: Protein flex coordinates of shape ``[F, M, 3]``.
            output_path: Path to save PDBQT file.
        """
        if torch.is_tensor(protein_coords):
            protein_coords = protein_coords.cpu().numpy()
        # Handles coordinate mapping from [F, M, 3] back to full pocket coordinates.
        self.protein_loader.save_pdbqt(output_path, flex_coordinates=protein_coords)


class VinaResultFormatProcessor(DockingResultProcessor):
    """Vina-format result processor.

    Generates docking results in AutoDock Vina output format, including
    MODEL/ENDMDL blocks, REMARK score lines, and RMSD annotations.
    """

    def process_and_save_results(
        self, selected_poses: list[dict], output_path: str
    ) -> list[float]:
        """Process selected poses and save to Vina-format PDBQT file.

        Args:
            selected_poses: Pose dictionaries from final evaluation.
                Each dict contains:
                ``'ligand_coords'`` -- ``[N_lig, 3]`` coordinates.
                ``'protein_coords'`` -- ``[N_prot, 3]`` flex residue coordinates.
                ``'total_score'`` -- INTER + INTRA energy.
                ``'inter_score'`` -- INTER energy.
                ``'ligand_intra_score'`` -- Ligand INTRA energy.
                ``'protein_intra_score'`` -- Protein INTRA energy.
            output_path: Path to output merged PDBQT file.

        Returns:
            List of scores: ``[torchdock_score, total_score, inter_score,
            intra_score, unbound_score]``.
        """
        if not selected_poses:
            self.logger.warning("No poses to save.")
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        self.logger.info(f"Processing {len(selected_poses)} selected poses...")

        # Create unique temporary directory per task to avoid race conditions.
        output_dir = os.path.dirname(output_path) or '.'
        unique_id = str(uuid.uuid4())[:8]
        temp_dir = os.path.join(output_dir, f'.temp_docking_results_{unique_id}')
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Calculate UNBOUND energy.
            # Vina/Vinardo: UNBOUND is top1's INTRA (shared by all poses).
            # ADFR: UNBOUND is each pose's own INTRA (calculated per-pose).
            unbound_energy = None
            if self.score_function in ["vina", "vinardo"]:
                unbound_energy = self._calculate_unbound_energy(selected_poses)

            # Calculate TorchDock scores and RMSDs.
            pose_data = self._calculate_torchdock_scores_and_rmsds(
                selected_poses, unbound_energy, temp_dir
            )

            # Generate Vina-format content.
            vina_content = self._generate_vina_format_content(pose_data, temp_dir)

            # Write to file.
            self._write_vina_format_file(vina_content, output_path)

            self.logger.info(f"Results saved to: {output_path}")

            # Extract best pose scores for return value.
            best_pose = pose_data[0]  # Already sorted by total_score.
            return [
                best_pose['torchdock_score'],
                best_pose['inter_intra'],
                best_pose['inter'],
                best_pose['intra'],
                best_pose['unbound'],
            ]

        finally:
            # Clean up temporary directory.
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    self.logger.debug(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {temp_dir}: {e}")

    def _calculate_torchdock_scores_and_rmsds(
        self,
        selected_poses: list[dict],
        unbound_energy: float | None,
        temp_dir: str,
    ) -> list[dict]:
        """Calculate TorchDock scores, RMSDs, and save temporary PDBQT files.

        TorchDock score formula:
            - Vina/Vinardo: ``((INTER + INTRA) - UNBOUND) / (1 + weight_torsion * num_torsions)``
            - ADFR: ``((INTER + INTRA) - UNBOUND) + weight_torsion * num_torsions``

        UNBOUND energy:
            - Vina/Vinardo: top1's INTRA (shared by all poses).
            - ADFR: each pose's own INTRA.

        RMSD calculation:
            - Top 1 pose (rank 1): rmsd_lb = rmsd_ub = 0.000 (reference).
            - Other poses: RMSD against top 1 pose (ignoring H atoms).

        Args:
            selected_poses: List of pose dictionaries.
            unbound_energy: UNBOUND energy for Vina/Vinardo. None for ADFR.
            temp_dir: Temporary directory for PDBQT files.

        Returns:
            Processed pose data with TorchDock scores and RMSDs.

        Raises:
            ValueError: If the scoring function is unsupported.
        """
        pose_data: list[dict] = []

        # Get reference structure (top 1) for RMSD calculation.
        ref_ligand_coords = selected_poses[0]['ligand_coords']
        atomicnums = self.ligand_loader.atomicnums
        adjacency_matrix = self.ligand_loader.adjacency_matrix

        for i, pose in enumerate(selected_poses):
            rank = i + 1

            # Extract scores.
            inter_intra = pose['total_score']  # INTER + INTRA
            inter = pose['inter_score']  # INTER
            intra = pose['ligand_intra_score'] + pose['protein_intra_score']  # INTRA

            # Calculate UNBOUND and TorchDock score based on scoring function.
            if self.score_function in ["vina", "vinardo"]:
                pose_unbound = unbound_energy
                torchdock_score = (
                    (inter_intra - pose_unbound)
                    / (1.0 + self.weight_torsion * self.degrees_of_freedom)
                )
            elif self.score_function in ["adfr", "ad4"]:
                pose_unbound = intra
                torchdock_score = (
                    (inter_intra - pose_unbound)
                    + self.weight_torsion * self.degrees_of_freedom
                )
            else:
                raise ValueError(f"Unsupported scoring function: {self.score_function}")

            # Calculate RMSD (ignore H atoms, use symmetric RMSD).
            if rank == 1:
                rmsd_lb = 0.0
                rmsd_ub = 0.0
            else:
                current_ligand_coords = pose['ligand_coords']
                rmsd_lb = calculate_rmsd(
                    current_ligand_coords,
                    ref_ligand_coords,
                    atomicnums,
                    atomicnums,
                    adjacency_matrix,
                    adjacency_matrix,
                    consider_symmetry=True,
                    ignore_hydrogen=True,
                )
                rmsd_ub = calculate_rmsd(
                    current_ligand_coords,
                    ref_ligand_coords,
                    atomicnums,
                    atomicnums,
                    adjacency_matrix,
                    adjacency_matrix,
                    consider_symmetry=False,
                    ignore_hydrogen=True,
                )

            # Save temporary PDBQT files.
            ligand_temp_path = os.path.join(temp_dir, f'ligand_rank_{rank}.pdbqt')
            self._save_ligand_pdbqt(pose['ligand_coords'], ligand_temp_path)

            pocket_temp_path = None
            if self.is_flexible_docking:
                pocket_temp_path = os.path.join(temp_dir, f'pocket_rank_{rank}.pdbqt')
                self._save_pocket_pdbqt(pose['protein_coords'], pocket_temp_path)

            pose_data.append({
                'rank': rank,
                'ligand_temp_path': ligand_temp_path,
                'pocket_temp_path': pocket_temp_path,
                'torchdock_score': torchdock_score,
                'inter_intra': inter_intra,
                'inter': inter,
                'intra': intra,
                'unbound': pose_unbound,
                'rmsd_lb': rmsd_lb,
                'rmsd_ub': rmsd_ub,
            })

        return pose_data

    def _generate_vina_format_content(
        self, pose_data: list[dict], temp_dir: str
    ) -> list[str]:
        """Generate Vina-format PDBQT content by reading temporary PDBQT files.

        Vina format structure::

            MODEL 1
            REMARK VINA RESULT: <torchdock_score> <rmsd_lb> <rmsd_ub>
            REMARK INTER + INTRA: <inter_intra>
            REMARK INTER:         <inter>
            REMARK INTRA:         <intra>
            REMARK UNBOUND:       <unbound>
            <ligand PDBQT content>
            [REMARK POCKET ATOMS START]
            [<pocket PDBQT content>]
            [REMARK POCKET ATOMS END]
            ENDMDL

        Args:
            pose_data: List of processed pose dictionaries.
            temp_dir: Temporary directory containing PDBQT files.

        Returns:
            Lines of Vina-format PDBQT content.
        """
        content_lines: list[str] = []

        for pose in pose_data:
            rank = pose['rank']

            # MODEL header.
            content_lines.append(f"MODEL {rank}\n")

            # REMARK lines with scores.
            content_lines.append(
                f"REMARK VINA RESULT: {pose['torchdock_score']:>12.3f} "
                f"{pose['rmsd_lb']:>12.3f} {pose['rmsd_ub']:>12.3f}\n"
            )
            content_lines.append(f"REMARK INTER + INTRA:        {pose['inter_intra']:>12.3f}\n")
            content_lines.append(f"REMARK INTER:                {pose['inter']:>12.3f}\n")
            content_lines.append(f"REMARK INTRA:                {pose['intra']:>12.3f}\n")
            content_lines.append(f"REMARK UNBOUND:              {pose['unbound']:>12.3f}\n")

            # Read ligand PDBQT content from temporary file.
            with open(pose['ligand_temp_path'], 'r') as f:
                ligand_content = f.read()
                if not ligand_content.endswith('\n'):
                    ligand_content += '\n'
                content_lines.append(ligand_content)

            # Flexible pocket PDBQT content (if flexible docking).
            if self.is_flexible_docking and pose['pocket_temp_path']:
                content_lines.append("REMARK POCKET ATOMS START\n")
                with open(pose['pocket_temp_path'], 'r') as f:
                    pocket_content = f.read()
                    if not pocket_content.endswith('\n'):
                        pocket_content += '\n'
                    content_lines.append(pocket_content)
                content_lines.append("REMARK POCKET ATOMS END\n")

            # MODEL footer.
            content_lines.append("ENDMDL\n")

        return content_lines

    def _write_vina_format_file(
        self, content_lines: list[str], output_path: str
    ) -> None:
        """Write Vina-format content to file.

        Args:
            content_lines: Lines to write.
            output_path: Output file path.
        """
        with open(output_path, 'w') as f:
            f.writelines(content_lines)
