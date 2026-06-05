"""
Conformer search strategies for molecular docking.

This module implements the core search loop for molecular docking optimization.
It provides an abstract base class and a SMAC-based implementation that combines
Bayesian optimization with gradient-based refinement to find optimal ligand poses.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from torchdock.metrics.rmsd import calculate_rmsd
from torchdock.pipeline.initializer import RandomInitializer, SMACInitializer
from torchdock.pipeline.score_predictor import ScorePredictor


class ConformerSearch(ABC):
    """Abstract base class for conformer search strategies.

    Subclasses must implement the :meth:`search` method to define
    how conformer optimization is performed.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.dtype = getattr(torch, config.dtype)
        self.device = torch.device(config.device)
        self.logger = self.config.logger

    @abstractmethod
    def search(self) -> list[dict]:
        """Run the conformer search and return selected poses.

        Returns:
            List of selected pose dictionaries.
        """
        raise NotImplementedError("Subclass must implement search method.")


class SMACConformerSearch(ConformerSearch):
    """SMAC-based conformer search with Bayesian optimization.

    Combines random forest surrogate modeling with gradient-based (Rprop)
    optimization to efficiently explore the conformational space. Uses
    Expected Improvement acquisition function for candidate selection
    and Monte Carlo mutation for local refinement.
    """

    def __init__(
        self,
        config: Any,
        ligand_loader: Any,
        protein_loader: Any,
        scoring_model: Any,
        ligand_transformer: Any,
        protein_transformer: Any,
    ) -> None:
        super().__init__(config)

        self.ligand_loader = ligand_loader
        self.protein_loader = protein_loader
        self.scoring_model = scoring_model
        self.ligand_transformer = ligand_transformer
        self.protein_transformer = protein_transformer

        self.random_initializer = RandomInitializer(config)
        self.smac_initializer = SMACInitializer(config, self.ligand_loader)
        self.score_predictor = ScorePredictor(config)

        # Rprop learning rates.
        self.rprop_lr_position: float = config.rprop_lr_position
        self.rprop_lr_orientation: float = config.rprop_lr_orientation
        self.rprop_lr_torsion: float = config.rprop_lr_torsion
        self.rprop_lr_chi: float = config.rprop_lr_chi

        # Search configuration.
        self.batch_size: int = config.batch_size
        self.max_restarts: int = config.max_restarts
        self.max_steps: int = config.max_steps
        self.update_interval: int = config.update_interval
        self.active_ratio_threshold: float = config.active_ratio_threshold
        self.convergence_tolerance: float = config.convergence_tolerance
        self.global_convergence_patience: int = config.global_convergence_patience

        # Output configuration.
        self.rmsd_threshold: float = config.rmsd_threshold
        self.num_poses: int = config.num_poses

        # Early stopping configuration.
        self.early_stop: bool = config.early_stop
        self.save_gradient: bool = config.save_gradient
        self.score_function: str = config.score_function

        # Number of early rounds that use random initialization to enhance diversity.
        self.num_random_init_rounds: int = 1 if self.max_restarts <= 5 else 2

        self.box_center = torch.tensor(config.box_center, dtype=self.dtype, device=self.device)
        self.box_size = torch.tensor(config.box_size, dtype=self.dtype, device=self.device)

        self.degrees_of_freedom: int = self.ligand_loader.degrees_of_freedom
        self.weight_torsion: float = self.scoring_model.weight_torsion.item()

        # Prepare ligand info for score predictor (heavy_atoms, num_torsions).
        total_atoms = self.ligand_loader.atoms_num
        num_hydrogens = len(self.ligand_loader.h_atom_indices)
        heavy_atoms = total_atoms - num_hydrogens
        num_torsions = len(self.ligand_loader.torsions)
        self.ligand_info: dict[str, int] = {
            'heavy_atoms': heavy_atoms,
            'num_torsions': num_torsions,
        }

        # Flag to indicate if pair cache needs to be rebuilt.
        self.need_rebuild_cache: bool = False

    def search(self) -> list[dict]:
        """Run the full SMAC-based conformer search loop.

        Iterates over multiple restart attempts, using either random or
        SMAC-based initialization, gradient optimization, and early
        stopping criteria.

        Returns:
            List of selected diverse poses with scores.
        """
        self._init_global_storage()
        smac_training_data = None

        for attempt in range(self.max_restarts):
            self._init_local_storage()
            self._init_conformer(attempt, smac_training_data)

            # Reset the optimizer on each attempt.
            optimizer = self._set_optimizer_parameters()

            # Clear pair cache at the start of each attempt.
            self.scoring_model.clear_pairs_cache()
            self.need_rebuild_cache = True

            pbar = tqdm(range(self.max_steps), desc=f"Attempt {attempt + 1}/{self.max_restarts}")
            for step in pbar:
                # Check if too few conformers are active.
                active_count = self.local_active_mask.sum()
                active_ratio = active_count / self.batch_size

                if active_ratio <= self.active_ratio_threshold:
                    break

                optimizer.zero_grad()

                # Update the ligand and protein coordinates.
                ligand_transformed_coords = self.ligand_transformer(self.local_active_mask)
                protein_transformed_coords = self.protein_transformer(self.local_active_mask)

                # Rebuild pair cache if needed (before scoring).
                if self.need_rebuild_cache:
                    self.scoring_model.rebuild_pairs_cache(ligand_transformed_coords, protein_transformed_coords)
                    self.need_rebuild_cache = False

                scores = self.scoring_model(ligand_transformed_coords, protein_transformed_coords, include_pocket_penalty=True)

                loss = scores[0].sum()
                loss.backward()

                if (step + 1) % self.update_interval == 0:
                    # Save gradient tracking if enabled (before mask update).
                    if (self.save_gradient or self.early_stop) and attempt == 0:
                        self._collect_gradients(step, scores)

                        # Predict convergence score if enabled.
                        if self.early_stop and (step + 1) // self.update_interval == 10:
                            predicted_score = self.score_predictor.predict(
                                self.epoch0_gradient_tracking,
                                self.ligand_info
                            )
                            self.logger.info(f"Predicted convergence score: {predicted_score}")
                            return predicted_score

                    # Save current parameters and scores for active samples.
                    old_mask = self.local_active_mask.clone()
                    self._save_local_history(scores, old_mask)

                    # Update active mask based on stopping conditions.
                    self.local_active_mask = self._update_active_mask_by_stop_conditions(
                        ligand_transformed_coords, old_mask
                    )

                    # Cache will be rebuilt with new active_mask coordinates.
                    self.need_rebuild_cache = True

                # Gradient clipping for position, orientation, and torsion parameters.
                torch.nn.utils.clip_grad_norm_(self.ligand_transformer.position_delta, 30)
                torch.nn.utils.clip_grad_norm_(self.ligand_transformer.orientation_quaternion, 20)
                torch.nn.utils.clip_grad_norm_(self.ligand_transformer.torsion_angles, 10)

                # Gradient clipping for chi angles.
                torch.nn.utils.clip_grad_norm_(self.protein_transformer.protein_torsion_angles, 10)

                optimizer.step()

            # Update global storage with local results.
            self._extract_final_conformers_from_local()

            # Visualize POT distribution for debugging (if enabled).
            if hasattr(self.config, 'debug_visualize_pot') and self.config.debug_visualize_pot:
                self._visualize_pot_distribution(attempt)

            # Check attempt-level early stopping.
            should_stop = self._check_attempt_early_stopping(attempt)
            if should_stop:
                self.logger.info(f"Early stopping triggered at attempt {attempt + 1}/{self.max_restarts}")
                break

            # Prepare training data for next attempt (if not the last one).
            if attempt < self.max_restarts - 1:
                smac_training_data = self.get_smac_training_data()

        return self._final_evaluation()

    def _init_global_storage(self) -> None:
        """Initialize global storage for final conformers, history, and gradient tracking."""
        # Global final conformers storage.
        self.global_final_conformers: dict[str, list] = {
            'pot_params': [],   # [N_total, 3 + 4 + T]
            'chi_params': [],   # [N_total, F, 4]
            'total_score': [],  # [N_total]
        }
        # Global history storage for SMAC training and analysis.
        self.global_history: dict[str, list] = {
            'pot_features': [],
            'chi_features': [],
            'final_total_score': [],
            'final_inter_score': [],
            'final_ligand_intra_score': [],
            'final_protein_intra_score': [],
        }

        # Track best score for each attempt (for attempt-level early stopping).
        self.attempt_best_scores: list[float] = []

        # First-attempt gradient and score tracking.
        self.epoch0_gradient_tracking: dict[str, list] = {
            'position_gradient': [],
            'orientation_gradient': [],
            'torsion_gradient': [],
            'torchdock_score': [],
            'total_score': [],
            'inter_score': [],
            'ligand_intra_score': [],
            'protein_intra_score': [],
            'cpu_time': [],
        }
        # Record CPU time start for gradient tracking.
        self._gradient_tracking_cpu_start = os.times()

        # Track last recorded scores for early-stopped conformers.
        self._last_recorded_scores: dict[str, torch.Tensor] | None = None

    def _init_local_storage(self) -> None:
        """Initialize local (per-attempt) storage tensors for POT, chi, and score histories."""
        self.max_history: int = (self.max_steps // self.update_interval) + 1

        num_torsions = self.ligand_transformer.torsion_count
        num_flexible_residues = self.protein_transformer.F
        float_nan = float('nan')

        # POT history: [batch_size, max_history, 3+4+T].
        pot_dim = 3 + 4 + num_torsions
        self.local_pot_history = torch.full(
            (self.batch_size, self.max_history, pot_dim),
            float_nan, dtype=self.dtype, device=self.device
        )

        # Chi history: [batch_size, max_history, F, 4].
        self.local_chi_history = torch.full(
            (self.batch_size, self.max_history, num_flexible_residues, 4),
            float_nan, dtype=self.dtype, device=self.device
        )

        # Score histories: [batch_size, max_history].
        self.local_total_score_history = torch.full(
            (self.batch_size, self.max_history), float_nan, dtype=self.dtype, device=self.device
        )
        self.local_inter_score_history = torch.full(
            (self.batch_size, self.max_history), float_nan, dtype=self.dtype, device=self.device
        )
        self.local_ligand_intra_score_history = torch.full(
            (self.batch_size, self.max_history), float_nan, dtype=self.dtype, device=self.device
        )
        self.local_protein_intra_score_history = torch.full(
            (self.batch_size, self.max_history), float_nan, dtype=self.dtype, device=self.device
        )

        # History write pointer.
        self.history_write_count = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        # Local active mask.
        self.local_active_mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)

    def _init_conformer(self, attempt: int, smac_training_data: dict | None) -> None:
        """Initialize conformers for the given attempt.

        Args:
            attempt: Current attempt index (0-based).
            smac_training_data: Training data from previous attempts. None for random init rounds.
        """
        if attempt < self.num_random_init_rounds:
            self.random_initializer.init_conformer(self.ligand_transformer, self.protein_transformer)
        else:
            self.smac_initializer.init_conformer(
                attempt, self.ligand_transformer, self.protein_transformer, smac_training_data
            )

    def _set_optimizer_parameters(self) -> torch.optim.Rprop:
        """Configure and return the Rprop optimizer for ligand and protein parameters.

        Returns:
            Rprop optimizer with per-parameter learning rates.
        """
        params = [
            {'params': self.ligand_transformer.position_delta, 'lr': self.rprop_lr_position},
            {'params': self.ligand_transformer.orientation_quaternion, 'lr': self.rprop_lr_orientation},
            {'params': self.ligand_transformer.torsion_angles, 'lr': self.rprop_lr_torsion},
            {'params': self.protein_transformer.protein_torsion_angles, 'lr': self.rprop_lr_chi},
        ]
        return torch.optim.Rprop(params)

    def _update_active_mask_by_stop_conditions(
        self, transformed_ligand_coords: torch.Tensor, active_mask: torch.Tensor
    ) -> torch.Tensor:
        """Check stopping conditions and return updated active mask.

        Args:
            transformed_ligand_coords: Coordinates of active samples, shape ``[K, N, 3]``.
            active_mask: Current active status, shape ``[B]``.

        Returns:
            Updated active mask with newly stopped samples set to False.
        """
        updated_active_mask = active_mask.clone()

        if not active_mask.any():
            return updated_active_mask

        active_indices = torch.where(active_mask)[0]
        expected_K = active_indices.numel()

        if transformed_ligand_coords.shape[0] != expected_K:
            raise ValueError(
                f"Shape mismatch: 'transformed_ligand_coords' has batch size "
                f"{transformed_ligand_coords.shape[0]}, but 'active_mask' indicates "
                f"{expected_K} active samples."
            )

        # Box boundary check.
        outside_global = self._check_box_boundary_condition(transformed_ligand_coords, active_indices)
        if outside_global.numel() > 0:
            updated_active_mask[outside_global] = False

        # Convergence check.
        converged_global = self._check_convergence_condition(
            active_indices, convergence_threshold=self.convergence_tolerance
        )
        if converged_global.numel() > 0:
            updated_active_mask[converged_global] = False

        return updated_active_mask

    def _check_box_boundary_condition(
        self, transformed_ligand_coords: torch.Tensor, active_indices: torch.Tensor, margin: float = 1.5
    ) -> torch.Tensor:
        """Check if ligand centroid is outside the expanded box (with margin).

        Args:
            transformed_ligand_coords: Coordinates of active samples, shape ``[K, N, 3]``.
            active_indices: Global indices of active samples, shape ``[K]``.
            margin: Extra tolerance in Angstroms. Defaults to 1.5.

        Returns:
            Global indices of samples outside the expanded box.
        """
        box_center = self.box_center
        box_size = self.box_size

        expanded_size = box_size + 2 * margin
        box_min = box_center - expanded_size / 2.0
        box_max = box_center + expanded_size / 2.0

        ligand_centroids = transformed_ligand_coords.mean(dim=1)

        inside_box = torch.all(
            (ligand_centroids >= box_min) & (ligand_centroids <= box_max),
            dim=1
        )

        return active_indices[~inside_box]

    def _check_convergence_condition(
        self, active_indices: torch.Tensor, convergence_threshold: float = 0.01
    ) -> torch.Tensor:
        """Check convergence based on recent score history (vectorized).

        A sample is considered converged if the last three recorded scores
        satisfy: |E_t - E_{t-1}| < threshold AND |E_{t-1} - E_{t-2}| < threshold.

        Args:
            active_indices: Global indices of active samples, shape ``[K]``.
            convergence_threshold: Threshold for energy change. Defaults to 0.01.

        Returns:
            Global indices of converged samples.
        """
        if len(active_indices) == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        write_counts = self.history_write_count[active_indices]

        has_enough_history = write_counts >= 3
        if not has_enough_history.any():
            return torch.empty(0, dtype=torch.long, device=self.device)

        score_hist = self.local_total_score_history[active_indices]

        idx_t = write_counts - 1
        idx_t1 = write_counts - 2
        idx_t2 = write_counts - 3

        valid_mask = has_enough_history
        K_range = torch.arange(len(active_indices), device=self.device)

        E_t = score_hist[K_range[valid_mask], idx_t[valid_mask]]
        E_t1 = score_hist[K_range[valid_mask], idx_t1[valid_mask]]
        E_t2 = score_hist[K_range[valid_mask], idx_t2[valid_mask]]

        delta1 = torch.abs(E_t - E_t1)
        delta2 = torch.abs(E_t1 - E_t2)

        converged_mask = (delta1 < convergence_threshold) & (delta2 < convergence_threshold)

        valid_active_indices = active_indices[valid_mask]
        return valid_active_indices[converged_mask]

    def _check_attempt_early_stopping(self, current_attempt: int) -> bool:
        """Check if attempt-level early stopping should be triggered.

        Early stopping triggers when no significant improvement over the global
        best score is achieved for ``global_convergence_patience`` consecutive
        attempts.

        Args:
            current_attempt: Current attempt index (0-based).

        Returns:
            True if early stopping should be triggered.
        """
        if not self.global_final_conformers['total_score']:
            self.logger.warning(f"Attempt {current_attempt + 1}: No conformers found.")
            return False

        current_best_score = self.global_final_conformers['total_score'][-1].min().item()
        self.attempt_best_scores.append(current_best_score)

        if not hasattr(self, '_global_best_score'):
            self._global_best_score: float = float('inf')
            self._last_improvement_attempt: int = -1

        global_best_old = self._global_best_score

        if current_best_score < self._global_best_score - self.convergence_tolerance:
            self._global_best_score = current_best_score
            self._last_improvement_attempt = current_attempt
            improved = True
        else:
            improved = False

        if current_attempt == 0:
            self.logger.info(
                f"Attempt {current_attempt + 1}/{self.max_restarts}: "
                f"Best score = {current_best_score:.4f} (initial attempt, global best)"
            )
        else:
            if improved:
                improvement = global_best_old - current_best_score
                self.logger.info(
                    f"Attempt {current_attempt + 1}/{self.max_restarts}: "
                    f"Best score = {current_best_score:.4f}, "
                    f"Improvement = {improvement:.4f} (new global best, previous: {global_best_old:.4f})"
                )
            else:
                patience_used = current_attempt - self._last_improvement_attempt
                self.logger.info(
                    f"Attempt {current_attempt + 1}/{self.max_restarts}: "
                    f"Best score = {current_best_score:.4f}, "
                    f"No significant improvement (patience: {patience_used}/{self.global_convergence_patience}, "
                    f"global best: {self._global_best_score:.4f})"
                )

        if current_attempt - self._last_improvement_attempt >= self.global_convergence_patience:
            self.logger.info(
                f"Attempt-level early stopping: No improvement for "
                f"{self.global_convergence_patience} consecutive attempts "
                f"(last improvement at attempt {self._last_improvement_attempt + 1})"
            )
            return True

        return False

    def _collect_gradients(self, step: int, scores: tuple) -> None:
        """Collect gradients and scores for the first attempt only.

        Collects data at update_interval points from 1 to 10 (e.g., steps 9, 19,
        ..., 99 when update_interval=10). For early-stopped conformers, scores
        are filled with last recorded values and gradients are set to 0.

        Args:
            step: Current step index (0-based).
            scores: Tuple of (total_score, inter_score, ligand_intra, protein_intra)
                tensors, each of shape ``[K]``.
        """
        collection_point = (step + 1) // self.update_interval
        if collection_point < 1 or collection_point > 10:
            return

        total_scores, inter_scores, ligand_intra_scores, protein_intra_scores = scores

        active_indices = torch.where(self.local_active_mask)[0]

        # Initialize last recorded scores on first collection.
        if collection_point == 1 and self._last_recorded_scores is None:
            self._last_recorded_scores = {
                'total_score': torch.full((self.batch_size,), float('nan'), dtype=self.dtype),
                'inter_score': torch.full((self.batch_size,), float('nan'), dtype=self.dtype),
                'ligand_intra_score': torch.full((self.batch_size,), float('nan'), dtype=self.dtype),
                'protein_intra_score': torch.full((self.batch_size,), float('nan'), dtype=self.dtype),
            }

        # Prepare full batch tensors.
        num_torsions = self.ligand_transformer.torsion_count
        position_grad_full = torch.zeros(self.batch_size, 3, dtype=self.dtype)
        orientation_grad_full = torch.zeros(self.batch_size, 4, dtype=self.dtype)
        torsion_grad_full = torch.zeros(self.batch_size, num_torsions, dtype=self.dtype)

        _nan = float('nan')
        total_score_full = self._last_recorded_scores['total_score'].clone() if self._last_recorded_scores else torch.full((self.batch_size,), _nan, dtype=self.dtype)
        inter_score_full = self._last_recorded_scores['inter_score'].clone() if self._last_recorded_scores else torch.full((self.batch_size,), _nan, dtype=self.dtype)
        ligand_intra_score_full = self._last_recorded_scores['ligand_intra_score'].clone() if self._last_recorded_scores else torch.full((self.batch_size,), _nan, dtype=self.dtype)
        protein_intra_score_full = self._last_recorded_scores['protein_intra_score'].clone() if self._last_recorded_scores else torch.full((self.batch_size,), _nan, dtype=self.dtype)

        if active_indices.numel() > 0:
            position_grad_active = self.ligand_transformer.position_delta.grad[active_indices].clone().detach().cpu()
            orientation_grad_active = self.ligand_transformer.orientation_quaternion.grad[active_indices].clone().detach().cpu()

            if self.ligand_transformer.torsion_angles.grad is not None:
                torsion_grad_active = self.ligand_transformer.torsion_angles.grad[active_indices].clone().detach().cpu()
            else:
                torsion_grad_active = torch.zeros(len(active_indices), 0)

            total_score_active = total_scores.clone().detach().cpu()
            inter_score_active = inter_scores.clone().detach().cpu()
            ligand_intra_score_active = ligand_intra_scores.clone().detach().cpu()
            protein_intra_score_active = protein_intra_scores.clone().detach().cpu()

            position_grad_full[active_indices] = position_grad_active
            orientation_grad_full[active_indices] = orientation_grad_active
            if num_torsions > 0:
                torsion_grad_full[active_indices] = torsion_grad_active

            total_score_full[active_indices] = total_score_active
            inter_score_full[active_indices] = inter_score_active
            ligand_intra_score_full[active_indices] = ligand_intra_score_active
            protein_intra_score_full[active_indices] = protein_intra_score_active

            self._last_recorded_scores['total_score'][active_indices] = total_score_active
            self._last_recorded_scores['inter_score'][active_indices] = inter_score_active
            self._last_recorded_scores['ligand_intra_score'][active_indices] = ligand_intra_score_active
            self._last_recorded_scores['protein_intra_score'][active_indices] = protein_intra_score_active

        # Compute TorchDock score for all batch samples.
        if self.score_function in ["vina", "vinardo"]:
            torchdock_score_full = inter_score_full / (1.0 + self.weight_torsion * self.degrees_of_freedom)
        elif self.score_function == "ad4":
            torchdock_score_full = inter_score_full + self.weight_torsion * self.degrees_of_freedom
        else:
            raise ValueError(f"Invalid score function: {self.score_function}")

        # Calculate CPU time elapsed since gradient tracking started.
        cpu_times_current = os.times()
        cpu_time_elapsed = (
            cpu_times_current.user - self._gradient_tracking_cpu_start.user
            + cpu_times_current.system - self._gradient_tracking_cpu_start.system
        )

        self.epoch0_gradient_tracking['position_gradient'].append(position_grad_full)
        self.epoch0_gradient_tracking['orientation_gradient'].append(orientation_grad_full)
        self.epoch0_gradient_tracking['torsion_gradient'].append(torsion_grad_full)
        self.epoch0_gradient_tracking['torchdock_score'].append(torchdock_score_full)
        self.epoch0_gradient_tracking['total_score'].append(total_score_full)
        self.epoch0_gradient_tracking['inter_score'].append(inter_score_full)
        self.epoch0_gradient_tracking['ligand_intra_score'].append(ligand_intra_score_full)
        self.epoch0_gradient_tracking['protein_intra_score'].append(protein_intra_score_full)
        self.epoch0_gradient_tracking['cpu_time'].append(cpu_time_elapsed)


    def _save_local_history(
        self, scores: tuple, old_active_mask: torch.Tensor
    ) -> None:
        """Save history for all samples active at the start of this step.

        Args:
            scores: Tuple of (total_scores, ligand_inter_scores,
                ligand_intra_scores, protein_intra_scores), each ``[K]``.
            old_active_mask: Boolean mask of active samples at step start, ``[B]``.
        """
        if not old_active_mask.any():
            return

        old_active_indices = torch.where(old_active_mask)[0]
        K = old_active_indices.numel()

        total_scores, ligand_inter_scores, ligand_intra_scores, protein_intra_scores = scores

        if total_scores.shape[0] != K:
            raise ValueError(
                f"Shape mismatch: 'total_score' has batch size {total_scores.shape[0]}, "
                f"but 'old_active_mask' indicates {K} active samples."
            )

        write_pos = self.history_write_count[old_active_indices]

        # Save ligand POT parameters.
        ligand_position = self.ligand_transformer.position_delta[old_active_indices]
        ligand_orientation = self.ligand_transformer.orientation_quaternion[old_active_indices]
        ligand_torsion = self.ligand_transformer.torsion_angles[old_active_indices]
        pot_params = torch.cat([ligand_position, ligand_orientation, ligand_torsion], dim=1)

        self.local_pot_history[old_active_indices, write_pos, :] = pot_params

        # Save protein chi angles.
        chi_params = self.protein_transformer.protein_torsion_angles[old_active_indices]
        self.local_chi_history[old_active_indices, write_pos, :, :] = chi_params

        # Save energy histories.
        self.local_total_score_history[old_active_indices, write_pos] = total_scores
        self.local_inter_score_history[old_active_indices, write_pos] = ligand_inter_scores
        self.local_ligand_intra_score_history[old_active_indices, write_pos] = ligand_intra_scores
        self.local_protein_intra_score_history[old_active_indices, write_pos] = protein_intra_scores

        # Update write counters.
        self.history_write_count[old_active_indices] += 1

    def _extract_final_conformers_from_local(self) -> None:
        """Extract the final (last valid) conformer for each batch index."""
        counts = self.history_write_count

        valid_mask = counts > 0
        if not valid_mask.any():
            return

        valid_indices = torch.where(valid_mask)[0]
        last_indices = counts[valid_mask] - 1

        pot_final = self.local_pot_history[valid_indices, last_indices]
        chi_final = self.local_chi_history[valid_indices, last_indices]
        total_score_final = self.local_total_score_history[valid_indices, last_indices]

        self.global_final_conformers['pot_params'].append(pot_final)
        self.global_final_conformers['chi_params'].append(chi_final)
        self.global_final_conformers['total_score'].append(total_score_final)

        # Extract all historical records for SMAC training.
        self._extract_history_for_smac_training(valid_indices, counts)

    def _extract_history_for_smac_training(
        self, valid_indices: torch.Tensor, counts: torch.Tensor
    ) -> None:
        """Extract all historical records for SMAC training.

        For each conformer, all historical POT and chi parameters are extracted
        as features, sharing the same label (the final converged score).

        Args:
            valid_indices: Batch indices with at least one history record, ``[M]``.
            counts: Number of history records for each batch index, ``[batch_size]``.
        """
        pot_features_list: list[torch.Tensor] = []
        chi_features_list: list[torch.Tensor] = []
        final_total_scores_list: list[torch.Tensor] = []
        final_inter_scores_list: list[torch.Tensor] = []
        final_ligand_intra_scores_list: list[torch.Tensor] = []
        final_protein_intra_scores_list: list[torch.Tensor] = []

        valid_counts = counts[valid_indices]

        for i, idx in enumerate(valid_indices):
            num_records = valid_counts[i].item()
            if num_records == 0:
                continue

            pot_history = self.local_pot_history[idx, :num_records, :]
            chi_history = self.local_chi_history[idx, :num_records, :, :]

            last_idx = num_records - 1
            final_total_score = self.local_total_score_history[idx, last_idx]
            final_inter_score = self.local_inter_score_history[idx, last_idx]
            final_ligand_intra_score = self.local_ligand_intra_score_history[idx, last_idx]
            final_protein_intra_score = self.local_protein_intra_score_history[idx, last_idx]

            pot_features_list.append(pot_history.detach().cpu())
            chi_features_list.append(chi_history.detach().cpu())

            final_total_scores_list.append(final_total_score.expand(num_records).clone().detach().cpu())
            final_inter_scores_list.append(final_inter_score.expand(num_records).clone().detach().cpu())
            final_ligand_intra_scores_list.append(final_ligand_intra_score.expand(num_records).clone().detach().cpu())
            final_protein_intra_scores_list.append(final_protein_intra_score.expand(num_records).clone().detach().cpu())

        if pot_features_list:
            self.global_history['pot_features'].extend(pot_features_list)
            self.global_history['chi_features'].extend(chi_features_list)
            self.global_history['final_total_score'].extend(final_total_scores_list)
            self.global_history['final_inter_score'].extend(final_inter_scores_list)
            self.global_history['final_ligand_intra_score'].extend(final_ligand_intra_scores_list)
            self.global_history['final_protein_intra_score'].extend(final_protein_intra_scores_list)

    def get_smac_training_data(self) -> dict[str, Any]:
        """Convert global history to numpy arrays for SMAC training.

        Should be called after each attempt to provide training data.

        Returns:
            Dictionary with keys:
                ``'pot_features'`` -- ``[N, 3+4+T]`` numpy array.
                ``'chi_features'`` -- ``[N, F, 4]`` numpy array.
                ``'labels'`` -- dict of score arrays.
                ``'num_samples'`` -- total number of training samples.
                ``'global_best_pot'`` -- ``[M, 3+4+T]`` best POT parameters.
                ``'global_best_scores'`` -- ``[M]`` best scores.
        """
        if not self.global_history['pot_features']:
            self.logger.warning("No training data available in global_history.")
            return {
                'pot_features': np.array([]),
                'chi_features': np.array([]),
                'labels': {
                    'total_score': np.array([]),
                    'inter_score': np.array([]),
                    'ligand_intra_score': np.array([]),
                    'protein_intra_score': np.array([]),
                },
                'num_samples': 0,
                'global_best_pot': np.array([]),
                'global_best_scores': np.array([]),
            }

        pot_features = torch.cat(self.global_history['pot_features'], dim=0).numpy()
        chi_features = torch.cat(self.global_history['chi_features'], dim=0).numpy()

        total_scores = torch.cat(self.global_history['final_total_score'], dim=0).numpy()
        inter_scores = torch.cat(self.global_history['final_inter_score'], dim=0).numpy()
        ligand_intra_scores = torch.cat(self.global_history['final_ligand_intra_score'], dim=0).numpy()
        protein_intra_scores = torch.cat(self.global_history['final_protein_intra_score'], dim=0).numpy()

        num_samples = pot_features.shape[0]

        # Extract global best conformers (sorted by total score).
        if self.global_final_conformers['pot_params']:
            global_pot = torch.cat(self.global_final_conformers['pot_params'], dim=0)
            global_scores = torch.cat(self.global_final_conformers['total_score'], dim=0)

            sorted_indices = torch.argsort(global_scores)
            top_k = min(self.batch_size, len(sorted_indices))
            global_best_pot = global_pot[sorted_indices[:top_k]].detach().cpu().numpy()
            global_best_scores = global_scores[sorted_indices[:top_k]].detach().cpu().numpy()
        else:
            global_best_pot = np.array([])
            global_best_scores = np.array([])

        return {
            'pot_features': pot_features,
            'chi_features': chi_features,
            'labels': {
                'total_score': total_scores,
                'inter_score': inter_scores,
                'ligand_intra_score': ligand_intra_scores,
                'protein_intra_score': protein_intra_scores,
            },
            'num_samples': num_samples,
            'global_best_pot': global_best_pot,
            'global_best_scores': global_best_scores,
        }

    def _final_evaluation(self) -> list[dict]:
        """Final evaluation and RMSD-based clustering of all collected conformers.

        Selects up to ``self.num_poses`` top-ranked diverse poses.

        Returns:
            List of selected diverse poses with parameters and scores.
        """
        if not self.global_final_conformers['pot_params']:
            self.logger.warning("No conformers collected for final evaluation.")
            return []

        pot_all = torch.cat(self.global_final_conformers['pot_params'], dim=0)
        chi_all = torch.cat(self.global_final_conformers['chi_params'], dim=0)
        scores_all = torch.cat(self.global_final_conformers['total_score'], dim=0)

        # Sort by total score (ascending: lower energy = better).
        sorted_indices = torch.argsort(scores_all)
        pot_all = pot_all[sorted_indices]
        chi_all = chi_all[sorted_indices]
        scores_all = scores_all[sorted_indices]

        N_total = pot_all.size(0)
        selected_poses: list[dict] = []
        selected_coords: list[torch.Tensor] = []

        # Verify dimensions.
        num_torsions = self.ligand_transformer.torsion_count
        num_flexible_residues = self.protein_transformer.F
        expected_pot_dim = 3 + 4 + num_torsions
        if pot_all.size(1) != expected_pot_dim:
            raise ValueError(
                f"POT dimension mismatch: expected {expected_pot_dim}, got {pot_all.size(1)}"
            )

        atomicnums = self.ligand_loader.atomicnums
        adjacency_matrix = self.ligand_loader.adjacency_matrix

        # Process in batches of self.batch_size.
        batch_start = 0
        while len(selected_poses) < self.num_poses and batch_start < N_total:
            batch_end = min(batch_start + self.batch_size, N_total)
            batch_size_actual = batch_end - batch_start

            pot_batch = pot_all[batch_start:batch_end]
            chi_batch = chi_all[batch_start:batch_end]

            position_delta = pot_batch[:, :3]
            orientation_quaternion = pot_batch[:, 3:7]
            torsion_angles = pot_batch[:, 7:]

            # Pad to full batch_size if needed (reset_parameters requires [batch_size, ...]).
            if batch_size_actual < self.batch_size:
                pad_size = self.batch_size - batch_size_actual
                position_delta = torch.cat([
                    position_delta,
                    torch.zeros(pad_size, 3, dtype=self.dtype, device=self.device)
                ], dim=0)
                orientation_quaternion = torch.cat([
                    orientation_quaternion,
                    torch.zeros(pad_size, 4, dtype=self.dtype, device=self.device)
                ], dim=0)
                orientation_quaternion[batch_size_actual:, 0] = 1.0
                torsion_angles = torch.cat([
                    torsion_angles,
                    torch.zeros(pad_size, num_torsions, dtype=self.dtype, device=self.device)
                ], dim=0)
                chi_batch = torch.cat([
                    chi_batch,
                    torch.zeros(pad_size, num_flexible_residues, 4, dtype=self.dtype, device=self.device)
                ], dim=0)

            self.ligand_transformer.reset_parameters({
                'position_delta': position_delta,
                'orientation_quaternion': orientation_quaternion,
                'torsion_angles': torsion_angles,
            })

            self.protein_transformer.reset_parameters({
                'protein_torsion_angles': chi_batch,
            })

            batch_active_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
            batch_active_mask[:batch_size_actual] = True

            with torch.no_grad():
                ligand_coords_batch = self.ligand_transformer(batch_active_mask)
                protein_coords_batch = self.protein_transformer(batch_active_mask)

                self.scoring_model.clear_pairs_cache()
                self.scoring_model.rebuild_pairs_cache(ligand_coords_batch, protein_coords_batch)
                scores = self.scoring_model(ligand_coords_batch, protein_coords_batch, include_pocket_penalty=True)
                total_scores, inter_scores, lig_intra, prot_intra = scores

            # Cluster within this batch using RMSD.
            for i in range(batch_size_actual):
                if len(selected_poses) >= self.num_poses:
                    break

                current_lig_coord = ligand_coords_batch[i]

                is_similar = False
                for ref_coord in selected_coords:
                    rmsd_val = calculate_rmsd(
                        current_lig_coord,
                        ref_coord,
                        atomicnums,
                        atomicnums,
                        adjacency_matrix,
                        adjacency_matrix,
                        consider_symmetry=True,
                        ignore_hydrogen=True,
                    )
                    if rmsd_val <= self.rmsd_threshold:
                        is_similar = True
                        break

                if not is_similar:
                    selected_poses.append({
                        'ligand_coords': current_lig_coord.cpu(),
                        'protein_coords': protein_coords_batch[i].cpu(),
                        'total_score': total_scores[i].item(),
                        'inter_score': inter_scores[i].item(),
                        'ligand_intra_score': lig_intra[i].item(),
                        'protein_intra_score': prot_intra[i].item(),
                    })
                    selected_coords.append(current_lig_coord.cpu())

            batch_start += self.batch_size

        self.scoring_model.clear_pairs_cache()

        self.logger.info(
            f"Final evaluation complete: selected {len(selected_poses)} / "
            f"{self.num_poses} diverse poses from {N_total} total conformers"
        )

        return selected_poses

    def _visualize_pot_distribution(self, attempt: int) -> None:
        """Visualize POT parameter distribution using PCA for the current attempt.

        Projects high-dimensional POT space to 2D, colored by convergence energy.

        Args:
            attempt: Current attempt index (0-based).
        """
        import matplotlib
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        matplotlib.use('Agg')
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['figure.dpi'] = 300

        counts = self.history_write_count
        valid_mask = counts > 0
        if not valid_mask.any():
            self.logger.warning(f"Attempt {attempt + 1}: No POT features to visualize.")
            return

        valid_indices = torch.where(valid_mask)[0]

        pot_features_list: list[torch.Tensor] = []
        final_total_scores_list: list[torch.Tensor] = []
        for idx in valid_indices:
            num_records = counts[idx].item()
            if num_records == 0:
                continue
            pot_history = self.local_pot_history[idx, :num_records, :]
            pot_features_list.append(pot_history.detach().cpu())
            final_score = self.local_total_score_history[idx, num_records - 1]
            final_total_scores_list.append(final_score.expand(num_records).clone().detach().cpu())

        if not pot_features_list:
            self.logger.warning(f"Attempt {attempt + 1}: No valid POT records to visualize.")
            return

        pot_features = torch.cat(pot_features_list, dim=0).numpy()
        total_scores = torch.cat(final_total_scores_list, dim=0).numpy()

        if pot_features.shape[0] < 2:
            self.logger.warning(f"Attempt {attempt + 1}: Insufficient data points for PCA.")
            return

        n_components = min(2, pot_features.shape[0], pot_features.shape[1])
        pca = PCA(n_components=n_components)
        pot_2d = pca.fit_transform(pot_features)

        fig, ax = plt.subplots(figsize=(4, 3.5), dpi=300)

        vmin, vmax = -15.0, 0.0
        cmap = plt.cm.coolwarm

        N = pot_features.shape[0]
        if N > 1500:
            alpha, s = 0.20, 8
        elif N > 500:
            alpha, s = 0.4, 12
        else:
            alpha, s = 0.7, 20

        scatter = ax.scatter(
            pot_2d[:, 0],
            pot_2d[:, 1] if pot_2d.shape[1] > 1 else np.zeros(N),
            c=total_scores,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=s,
            alpha=alpha,
            edgecolors='none',
            rasterized=True,
        )

        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Convergence Energy (kcal/mol)', fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        variance_explained = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({variance_explained[0] * 100:.1f}%)', fontsize=10)
        if pot_2d.shape[1] > 1:
            ax.set_ylabel(f'PC2 ({variance_explained[1] * 100:.1f}%)', fontsize=10)
        else:
            ax.set_ylabel('Constant (0)', fontsize=10)

        ax.set_title(f'POT Distribution - Attempt {attempt + 1}', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=9)

        plt.tight_layout()

        output_dir = (
            os.path.dirname(self.config.output_file_path)
            if hasattr(self.config, 'output_file_path')
            else './'
        )
        os.makedirs(output_dir, exist_ok=True)

        output_filename = os.path.join(output_dir, f'pot_distribution_attempt_{attempt + 1:02d}.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        self.logger.info(
            f"Attempt {attempt + 1}: POT distribution saved to {output_filename} "
            f"(N={pot_features.shape[0]}, PCA variance: {variance_explained.sum() * 100:.1f}%)"
        )