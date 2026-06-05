"""
Conformer initialization strategies for molecular docking.

This module provides abstract and concrete initializers for generating
ligand conformations (position, orientation, torsion angles) before
gradient-based optimization. It includes random initialization and
SMAC-based Bayesian optimization initialization.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from ConfigSpace import ConfigurationSpace, Float
from smac.acquisition.function.expected_improvement import EI
from smac.model.random_forest.random_forest import RandomForest


class ConformerInitializer(ABC):
    """Abstract base class for conformer initialization strategies.

    Subclasses must implement the ``init_conformer`` method to define
    how ligand (and optionally protein) conformations are initialized
    before gradient-based docking optimization.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)

    @abstractmethod
    def init_conformer(self) -> None:
        """Initialize ligand and protein conformations.

        Subclasses must override this method with their specific
        initialization logic.
        """
        raise NotImplementedError("Subclass must implement init_conformer method.")


class RandomInitializer(ConformerInitializer):
    """Fully random conformer initializer.

    Samples ligand position uniformly within the docking box, orientation
    as a uniform random unit quaternion, and torsion angles from a wrapped
    Gaussian distribution. Protein side chains are kept at native
    conformation (zeros).
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.batch_size: int = config.batch_size

        self.box_center = torch.tensor(config.box_center, device=self.device, dtype=self.dtype)
        self.box_size = torch.tensor(config.box_size, device=self.device, dtype=self.dtype)

    def init_conformer(
        self,
        ligand_transform_module: Any,
        protein_transform_module: Any | None = None,
    ) -> None:
        """Fully random initialization of ligand conformations.

        Protein side chains are kept at native conformation (zeros).

        Args:
            ligand_transform_module: Ligand parameter transform module.
            protein_transform_module: Protein parameter transform module.
                Defaults to None.
        """
        # Compute box bounds.
        box_min = self.box_center - self.box_size / 2.0
        box_max = self.box_center + self.box_size / 2.0

        # --- Ligand: P (position_delta) ---
        # Sample absolute positions uniformly in the box.
        abs_positions = torch.rand(self.batch_size, 3, device=self.device, dtype=self.dtype) * (box_max - box_min) + box_min
        # Convert to delta relative to ligand's initial center.
        position_delta = abs_positions - ligand_transform_module.position_center_coords  # [batch, 3]

        # --- Ligand: O (orientation_quaternion) ---
        # Uniform random unit quaternion.
        orientation_quaternion = self._random_unit_quaternion(self.batch_size)

        # --- Ligand: T (torsion_angles) ---
        # Gaussian distribution with higher probability within +/-45 degrees.
        gaussian_std = torch.pi / 4.0
        torsion_angles = torch.randn(
            self.batch_size, ligand_transform_module.torsion_count,
            device=self.device, dtype=self.dtype
        ) * gaussian_std  # [batch, n_torsions]
        # Wrap angles to [-pi, pi] range.
        torsion_angles = torch.atan2(torch.sin(torsion_angles), torch.cos(torsion_angles))

        # Set ligand parameters.
        ligand_transform_module.reset_parameters({
            'position_delta': position_delta,
            'orientation_quaternion': orientation_quaternion,
            'torsion_angles': torsion_angles
        })

        # --- Protein: keep native (all zeros) ---
        if protein_transform_module is not None:
            protein_transform_module.reset_parameters({
                'protein_torsion_angles': torch.zeros(
                    self.batch_size, protein_transform_module.F, 4,
                    device=self.device, dtype=self.dtype
                )
            })

    def _random_unit_quaternion(self, batch_size: int) -> torch.Tensor:
        """Generate uniform random unit quaternions using Shoemake's method.

        Args:
            batch_size: Number of quaternions to generate.

        Returns:
            Tensor of shape ``[batch_size, 4]`` with format ``[w, x, y, z]``.
        """
        u1 = torch.rand(batch_size, device=self.device, dtype=self.dtype)
        u2 = torch.rand(batch_size, device=self.device, dtype=self.dtype)
        u3 = torch.rand(batch_size, device=self.device, dtype=self.dtype)

        q_w = torch.sqrt(1.0 - u1) * torch.sin(2 * torch.pi * u2)
        q_x = torch.sqrt(1.0 - u1) * torch.cos(2 * torch.pi * u2)
        q_y = torch.sqrt(u1) * torch.sin(2 * torch.pi * u3)
        q_z = torch.sqrt(u1) * torch.cos(2 * torch.pi * u3)

        return torch.stack([q_w, q_x, q_y, q_z], dim=1)


class SMACInitializer(ConformerInitializer):
    """SMAC-based conformer initializer using Bayesian optimization.

    Trains a random forest surrogate model on historical docking data
    and uses Expected Improvement (EI) acquisition function with Monte
    Carlo mutation to sample promising conformations.
    """

    def __init__(self, config: Any, ligand_loader: Any) -> None:
        super().__init__(config)
        self.batch_size: int = config.batch_size
        self.max_torsions: int = config.ligand_max_torsions

        self.box_center = np.array(config.box_center)
        self.box_size = np.array(config.box_size)

        # Number of random configurations for initial broad exploration.
        self.n_samples: int = int(self.batch_size * 10)
        self.torsion_count: int = min(self.max_torsions, len(ligand_loader.torsions))
        self.torsion_masks = ligand_loader.torsion_masks
        self.h_atom_indices = ligand_loader.h_atom_indices
        self.reference_coords = ligand_loader.ligand_coords  # Molecular reference coordinates.
        self.atom_center_coords = ligand_loader.atom_center_coords  # Molecular rotation center.
        self.position_center_coords = ligand_loader.position_center_coords  # Molecular geometric center.

        self.mutation_iterations: int = 100
        self.position_mutation_radius: float = 2.0  # P parameter mutation radius (Angstrom).
        self.molecular_size_factor: float = 2.0  # Used to calculate O parameter rotation amplitude.

        self.molecular_radius: float = self._calculate_molecular_radius()
        self.pot_probabilities, self.torsion_probabilities = self._initialize_pot_weights()

        self.acquisition_function = EI()

        self.surrogate_model = RandomForest(
            configspace=self.configspace,
            n_trees=150,
            ratio_features=0.33,
            min_samples_split=4,
            min_samples_leaf=3,
            # max_depth=5,
        )

    @property
    def configspace(self) -> ConfigurationSpace:
        """Define the configuration space for SMAC-based initialization.

        Includes ligand Position (P), Orientation (O), and Torsion (T)
        parameters.

        Returns:
            ConfigurationSpace with POT parameter definitions.
        """
        self.box_min = self.box_center - self.box_size / 2
        self.box_max = self.box_center + self.box_size / 2

        self.position_min = self.box_min - self.position_center_coords
        self.position_max = self.box_max - self.position_center_coords

        cs = ConfigurationSpace()

        # Position (P) parameters.
        position_delta_x = Float("1_position_delta_x", (self.position_min[0], self.position_max[0]))
        position_delta_y = Float("1_position_delta_y", (self.position_min[1], self.position_max[1]))
        position_delta_z = Float("1_position_delta_z", (self.position_min[2], self.position_max[2]))
        cs.add([position_delta_x, position_delta_y, position_delta_z])

        # Orientation (O) parameters (quaternion components).
        # Each quaternion component ranges [-1, 1], but will be normalized later.
        orientation_quaternion_w = Float("2_orientation_quaternion_w", (-1.0, 1.0))
        orientation_quaternion_x = Float("2_orientation_quaternion_x", (-1.0, 1.0))
        orientation_quaternion_y = Float("2_orientation_quaternion_y", (-1.0, 1.0))
        orientation_quaternion_z = Float("2_orientation_quaternion_z", (-1.0, 1.0))
        cs.add([orientation_quaternion_w, orientation_quaternion_x, orientation_quaternion_y, orientation_quaternion_z])

        # Torsion (T) parameters.
        # Each torsion angle ranges [-pi, pi].
        for i in range(self.torsion_count):
            torsion_angle = Float(f"3_torsion_angle_{i}", (-math.pi, math.pi), default=0.0)
            cs.add([torsion_angle])

        return cs

    def _calculate_molecular_radius(self) -> float:
        """Calculate the molecular radius of gyration.

        Used to determine the rotation amplitude of O parameters.
        Formula: Rg = sqrt(sum((ri - r0)^2) / N), where ri are heavy atom
        coordinates, r0 is the rotation center, and N is the number of
        heavy atoms.

        Returns:
            Molecular radius of gyration in Angstroms.
        """
        # Create a mask to exclude hydrogen atoms.
        heavy_atom_mask = np.ones(len(self.reference_coords), dtype=bool)
        if len(self.h_atom_indices) > 0:
            h_indices_list = list(self.h_atom_indices)
            heavy_atom_mask[h_indices_list] = False

        # Get coordinates of heavy atoms only.
        heavy_atom_coords = self.reference_coords[heavy_atom_mask]

        # Calculate the distance squared from each heavy atom to the rotation center.
        distances_squared = np.sum((heavy_atom_coords - self.atom_center_coords) ** 2, axis=1)

        # Calculate the radius of gyration: Rg = sqrt(sum(d^2) / N).
        n_heavy_atoms = len(heavy_atom_coords)
        rg = float(np.sqrt(np.sum(distances_squared) / n_heavy_atoms))

        return rg

    def _initialize_pot_weights(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Initialize POT (Position, Orientation, Torsion) weights and probabilities.

        Calculates POT probabilities based on parameter availability and
        torsion probabilities based on the number of affected heavy atoms.

        Returns:
            A tuple of (pot_probabilities, torsion_probabilities):
                - pot_probabilities: Array of shape ``[3]`` for P, O, T.
                - torsion_probabilities: Array of shape ``[torsion_count]``
                  or None if no torsions exist.

        Raises:
            ValueError: If a torsion only affects hydrogen atoms.
        """
        # Dynamically set weights for POT based on availability.
        p_weight = 1.0  # Position always available.
        o_weight = 1.0  # Orientation always available.
        t_weight = float(self.torsion_count)  # Torsion weight proportional to count.

        pot_weights = np.array([p_weight, o_weight, t_weight])
        pot_probabilities = pot_weights / pot_weights.sum()

        # Calculate torsion weights based on the number of affected heavy atoms.
        torsion_probabilities = None
        if self.torsion_count > 0:
            # Create heavy atom mask.
            heavy_atom_mask = np.ones(len(self.reference_coords), dtype=bool)
            if len(self.h_atom_indices) > 0:
                h_indices_list = list(self.h_atom_indices)
                heavy_atom_mask[h_indices_list] = False

            # Calculate affected heavy atoms count for each torsion.
            torsion_affected_heavy_atoms_count = np.zeros(self.torsion_count)
            for i in range(self.torsion_count):
                affected_atoms_mask = self.torsion_masks[i]  # [N_atoms]
                affected_heavy_atoms = affected_atoms_mask & heavy_atom_mask
                heavy_count = affected_heavy_atoms.sum()

                if heavy_count == 0:
                    raise ValueError(
                        f"PDBQT format error: Torsion {i} only affects hydrogen atoms. "
                        f"Valid torsions must affect at least one heavy atom."
                    )

                torsion_affected_heavy_atoms_count[i] = heavy_count

            # Calculate probabilities based on affected heavy atoms count.
            torsion_probabilities = torsion_affected_heavy_atoms_count / torsion_affected_heavy_atoms_count.sum()

        return pot_probabilities, torsion_probabilities

    def _preprocess_training_data(
        self, pot_features: np.ndarray, total_score: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Preprocess POT features and labels for SMAC training.

        Feature preprocessing:
            1. Normalize O (orientation quaternion) to unit quaternion.
            2. Wrap T (torsion angles) to [-pi, pi] range.

        Label preprocessing:
            1. Clip total_score to remove outliers.

        Args:
            pot_features: Feature array of shape ``[N, D]``.
            total_score: Score array of shape ``[N]``.

        Returns:
            A tuple of (processed_pot_features, processed_score, best_score):
                - processed_pot_features: Array of shape ``[N, D]``, float64.
                - processed_score: Array of shape ``[N]``, float64.
                - best_score: Minimum score after preprocessing.
        """
        pot_dim = 3 + 4 + self.torsion_count

        # Extract P, O, T components.
        position = pot_features[:, :3]
        orientation = pot_features[:, 3:7]
        torsion = pot_features[:, 7:pot_dim]

        # Normalize Orientation (O) to unit quaternion.
        quat_norm = np.linalg.norm(orientation, axis=1, keepdims=True)
        quat_norm = np.clip(quat_norm, a_min=1e-8, a_max=None)
        orientation_normalized = orientation / quat_norm

        # Wrap Torsion angles (T) to [-pi, pi].
        torsion_wrapped = np.mod(torsion + np.pi, 2 * np.pi) - np.pi

        # Reassemble preprocessed POT features.
        processed_pot_features = np.concatenate([position, orientation_normalized, torsion_wrapped], axis=1)

        # Clip total_score to remove outliers.
        best_score = total_score.min()
        processed_score = np.clip(total_score, a_min=None, a_max=best_score + 20.0)

        # Update best_score after clipping to ensure consistency.
        best_score = float(processed_score.min())

        # Ensure data types are correct for SMAC (float64).
        processed_pot_features = np.ascontiguousarray(processed_pot_features, dtype=np.float64)
        processed_score = np.ascontiguousarray(processed_score, dtype=np.float64)

        # Ensure processed_score is 1D.
        if processed_score.ndim != 1:
            processed_score = processed_score.ravel()

        return processed_pot_features, processed_score, best_score

    def _sample_configurations(self, n_samples: int) -> np.ndarray:
        """Sample configurations from the configuration space using pure NumPy.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Array of shape ``[n_samples, 3+4+T]`` with dtype float64,
            containing position delta (P), normalized quaternion (O),
            and torsion angles in [-pi, pi] (T).
        """
        feature_dim = 3 + 4 + self.torsion_count
        samples = np.zeros((n_samples, feature_dim))

        # Position parameters (P): random positions within box.
        abs_positions = np.random.uniform(self.box_min, self.box_max, size=(n_samples, 3))
        samples[:, 0:3] = abs_positions - self.position_center_coords

        # Orientation parameters (O): random unit quaternions [w, x, y, z].
        random_quat = np.random.randn(n_samples, 4)
        quat_norm = np.linalg.norm(random_quat, axis=1, keepdims=True)
        samples[:, 3:7] = random_quat / quat_norm

        # Torsion parameters (T): Gaussian distribution with std = pi/4.
        if self.torsion_count > 0:
            gaussian_std = np.pi / 4.0
            torsion_angles = np.random.randn(n_samples, self.torsion_count) * gaussian_std
            samples[:, 7:7 + self.torsion_count] = np.arctan2(np.sin(torsion_angles), np.cos(torsion_angles))

        return samples

    def mutate_position_parameters(
        self, position_params: np.ndarray, mutation_radius: float
    ) -> np.ndarray:
        """Mutate position parameters within a spherical region.

        Args:
            position_params: Position array of shape ``[batch_size, 3]``.
            mutation_radius: Mutation radius in Angstroms.

        Returns:
            Mutated position array of shape ``[batch_size, 3]``, clipped
            to box bounds.
        """
        batch_size = position_params.shape[0]

        # Generate standard normal random directions (spherical symmetry).
        random_directions = np.random.randn(batch_size, 3)
        norms = np.linalg.norm(random_directions, axis=1, keepdims=True)
        random_directions = np.divide(random_directions, norms, out=random_directions, where=norms != 0)

        # Spherical uniform distribution of radii: r ~ U(0, R^3)^(1/3).
        random_radii = mutation_radius * (np.random.random(batch_size) ** (1 / 3))

        # Construct displacement vectors.
        mutation_vectors = random_directions * random_radii[:, None]

        # Apply mutation.
        mutated_params = position_params + mutation_vectors

        # Clip to box boundaries to ensure valid positions.
        mutated_params = np.clip(mutated_params, self.position_min, self.position_max)

        return mutated_params

    def mutate_orientation_parameters(
        self, orientation_params: np.ndarray, molecular_radius: float
    ) -> np.ndarray:
        """Mutate orientation parameters (quaternions) based on molecular size.

        Args:
            orientation_params: Quaternion array of shape ``[batch_size, 4]``
                in ``[w, x, y, z]`` format.
            molecular_radius: Molecular radius of gyration in Angstroms.

        Returns:
            Mutated quaternion array of shape ``[batch_size, 4]``.
        """
        batch_size = orientation_params.shape[0]

        # Calculate the maximum rotation amplitude (radians).
        max_rotation_radians = self.molecular_size_factor / (molecular_radius + 1e-6)

        # Sample rotation vectors uniformly within a sphere.
        rand_dir = np.random.randn(batch_size, 3)
        rand_dir_norm = np.linalg.norm(rand_dir, axis=1, keepdims=True)
        rand_dir = rand_dir / rand_dir_norm

        # Random radii: r = R * u^(1/3) for uniform distribution within sphere.
        u = np.random.random(batch_size)
        rand_radii = max_rotation_radians * (u ** (1 / 3))

        rotation_vectors = rand_dir * rand_radii[:, None]

        # Convert rotation vectors to quaternions.
        theta = np.linalg.norm(rotation_vectors, axis=1)
        half_theta = theta / 2
        sin_half = np.sin(half_theta)
        cos_half = np.cos(half_theta)

        # Handle theta ~ 0 case (avoid division by zero).
        small_angle = (theta < 1e-6)
        rotation_quats = np.empty((batch_size, 4))
        rotation_quats[:, 0] = cos_half

        axis = np.divide(
            rotation_vectors, theta[:, None],
            where=theta[:, None] != 0,
            out=np.zeros_like(rotation_vectors),
        )
        rotation_quats[:, 1:] = sin_half[:, None] * axis

        # For zero angles: directly set to identity quaternion.
        rotation_quats[small_angle, 1:] = 0.0

        # Quaternion multiplication: q_new = q_orig * q_rot.
        q_orig = orientation_params / np.linalg.norm(orientation_params, axis=1, keepdims=True)
        w0, x0, y0, z0 = q_orig.T
        w1, x1, y1, z1 = rotation_quats.T

        q_new = np.stack([
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ], axis=-1)

        # Normalize.
        q_new /= np.linalg.norm(q_new, axis=1, keepdims=True)
        return q_new

    def mutate_angle_parameters(
        self, angles: np.ndarray, perturbation_scale: float = 0.3
    ) -> np.ndarray:
        """Mutate torsion angles with wrapped Gaussian noise.

        Args:
            angles: Angle array in radians, arbitrary shape.
            perturbation_scale: Standard deviation of Gaussian noise
                in radians. Defaults to 0.3 (~17 degrees).

        Returns:
            Mutated angles in [-pi, pi], same shape as input.
        """
        noise = np.random.normal(scale=perturbation_scale, size=angles.shape)
        new_angles = angles + noise
        # Wrap to [-pi, pi].
        return (new_angles + np.pi) % (2 * np.pi) - np.pi

    def monte_carlo_acceptance_batch(
        self,
        current_ei_values: np.ndarray,
        new_ei_values: np.ndarray,
        acceptance_probability: float = 1.0,
    ) -> np.ndarray:
        """Batch Monte Carlo acceptance criterion.

        Accepts with given probability when EI improves; never accepts
        when EI worsens.

        Args:
            current_ei_values: Current EI values of shape ``[batch_size]``.
            new_ei_values: New EI values of shape ``[batch_size]``.
            acceptance_probability: Probability of accepting when EI
                improves. Defaults to 1.0.

        Returns:
            Boolean array of shape ``[batch_size]`` indicating acceptance.
        """
        improvement = new_ei_values - current_ei_values
        better_mask = improvement > 0
        acceptance = np.zeros_like(better_mask, dtype=bool)

        if np.any(better_mask):
            random_vals = np.random.random(np.sum(better_mask))
            acceptance[better_mask] = random_vals < acceptance_probability

        return acceptance

    def _monte_carlo_mutation(
        self, features: np.ndarray, acquisition_values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform Monte Carlo mutation on POT features to optimize acquisition function.

        Args:
            features: Initial features of shape ``[n_preselect, feature_dim]``.
            acquisition_values: Initial acquisition values of shape ``[n_preselect]``.

        Returns:
            A tuple of (optimized_features, optimized_ei_values):
                - optimized_features: Array of shape ``[n_preselect, feature_dim]``.
                - optimized_ei_values: Array of shape ``[n_preselect]``.
        """
        current_features = features.copy()
        current_ei_values = acquisition_values.flatten()

        for _ in range(self.mutation_iterations):
            batch_size = current_features.shape[0]

            # Choose which parameter group to mutate for each sample.
            group_choices = np.random.choice(3, size=batch_size, p=self.pot_probabilities)

            # --- Position (P) ---
            p_mask = (group_choices == 0)
            if np.any(p_mask):
                p_indices = np.where(p_mask)[0]
                mutated_p = self.mutate_position_parameters(
                    current_features[p_indices, 0:3], self.position_mutation_radius
                )
                mutated_features_p = current_features.copy()
                mutated_features_p[p_indices, 0:3] = mutated_p
                new_ei_p = self.acquisition_function._compute(mutated_features_p[p_indices]).flatten()
                accept_p = self.monte_carlo_acceptance_batch(current_ei_values[p_indices], new_ei_p)
                update_idx = p_indices[accept_p]
                current_features[update_idx, 0:3] = mutated_p[accept_p]
                current_ei_values[update_idx] = new_ei_p[accept_p]

            # --- Orientation (O) ---
            o_mask = (group_choices == 1)
            if np.any(o_mask):
                o_indices = np.where(o_mask)[0]
                mutated_o = self.mutate_orientation_parameters(
                    current_features[o_indices, 3:7], self.molecular_radius
                )
                mutated_features_o = current_features.copy()
                mutated_features_o[o_indices, 3:7] = mutated_o
                new_ei_o = self.acquisition_function._compute(mutated_features_o[o_indices]).flatten()
                accept_o = self.monte_carlo_acceptance_batch(current_ei_values[o_indices], new_ei_o)
                update_idx = o_indices[accept_o]
                current_features[update_idx, 3:7] = mutated_o[accept_o]
                current_ei_values[update_idx] = new_ei_o[accept_o]

            # --- Torsions (T) ---
            if self.torsion_count > 0:
                t_mask = (group_choices == 2)
                if np.any(t_mask):
                    t_indices = np.where(t_mask)[0]
                    n_t_selected = len(t_indices)
                    selected_torsion_ids = np.random.choice(
                        self.torsion_count, size=n_t_selected, p=self.torsion_probabilities
                    )

                    feature_cols = 7 + selected_torsion_ids

                    current_angles = current_features[t_indices, feature_cols]
                    mutated_angles = self.mutate_angle_parameters(current_angles)

                    mutated_features_t = current_features.copy()
                    mutated_features_t[t_indices, feature_cols] = mutated_angles
                    new_ei_t = self.acquisition_function._compute(mutated_features_t[t_indices]).flatten()
                    accept_t = self.monte_carlo_acceptance_batch(current_ei_values[t_indices], new_ei_t)

                    if np.any(accept_t):
                        update_indices = t_indices[accept_t]
                        update_columns = feature_cols[accept_t]
                        current_features[update_indices, update_columns] = mutated_angles[accept_t]
                        current_ei_values[update_indices] = new_ei_t[accept_t]

        return current_features, current_ei_values

    def init_conformer(
        self,
        attempt: int,
        ligand_transform_module: Any,
        protein_transform_module: Any | None,
        training_data: dict[str, Any],
    ) -> None:
        """Initialize conformers using SMAC-based sampling from historical data.

        Sampling strategy:
            1. Generate random configurations and evaluate with acquisition function.
            2. Select top batch_size candidates by EI score.
            3. Replace a portion of candidates with global best POT parameters.
            4. Apply Monte Carlo mutation to optimize candidates.
            5. Set final parameters for optimization.

        Args:
            attempt: Current docking attempt number (used for replacement ratio).
            ligand_transform_module: Ligand parameter transform module.
            protein_transform_module: Protein parameter transform module.
                Defaults to None.
            training_data: Dictionary containing historical docking data with keys:
                - ``'pot_features'``: Historical POT features for training.
                - ``'labels'``: Score labels dictionary.
                - ``'num_samples'``: Number of historical samples.
                - ``'global_best_pot'``: Sorted global best POT parameters.
                - ``'global_best_scores'``: Corresponding scores.
        """
        self.training_data = training_data
        self.labels = training_data['labels']
        self.num_samples = training_data['num_samples']
        self.current_attempt = attempt

        pot_features = training_data['pot_features']
        total_score = self.labels['total_score']

        # Preprocess features and labels.
        features, labels, best_score = self._preprocess_training_data(pot_features, total_score)

        # Train surrogate model.
        self.surrogate_model.train(features, labels)

        # Update the acquisition function.
        self.acquisition_function.update(self.surrogate_model, eta=best_score)

        # Generate a large number of random features.
        random_features = self._sample_configurations(self.n_samples)

        # Evaluate the potential of the random features.
        acquisition_values = self.acquisition_function._compute(random_features)

        # Select top batch_size candidates by EI score.
        ei_scores = acquisition_values.ravel()
        top_k_indices = np.argpartition(ei_scores, -self.batch_size)[-self.batch_size:]
        top_k_features = random_features[top_k_indices]
        top_k_ei_scores = ei_scores[top_k_indices]

        # Replace a portion with global best POT parameters.
        global_best_pot = training_data.get('global_best_pot', np.array([]))

        if global_best_pot.size > 0 and hasattr(self, 'current_attempt'):
            # Calculate replacement ratio: current_attempt * 0.1, capped at 0.7.
            replacement_ratio = min(self.current_attempt * 0.1, 0.7)
            num_replace = int(self.batch_size * replacement_ratio)

            if num_replace > 0:
                num_available = min(num_replace, len(global_best_pot))
                best_pot_to_insert = global_best_pot[:num_available].astype(np.float64)

                # Replace the worst EI candidates with global best POT.
                worst_indices = np.argpartition(top_k_ei_scores, num_available)[:num_available]
                top_k_features[worst_indices] = best_pot_to_insert

                # Recalculate EI for replaced candidates.
                top_k_ei_scores[worst_indices] = self.acquisition_function._compute(best_pot_to_insert).flatten()

        # Optimize the candidates using Monte Carlo mutation.
        optimized_features, optimized_ei_scores = self._monte_carlo_mutation(top_k_features, top_k_ei_scores)

        # Extract POT parameters from final features.
        final_features = optimized_features
        position_delta = final_features[:, 0:3]
        orientation_quaternion = final_features[:, 3:7]
        torsion_angles = final_features[:, 7:7 + self.torsion_count]

        # Convert to PyTorch tensors and move to device.
        position_delta = torch.from_numpy(position_delta).to(device=self.device, dtype=self.dtype)
        orientation_quaternion = torch.from_numpy(orientation_quaternion).to(device=self.device, dtype=self.dtype)
        torsion_angles = torch.from_numpy(torsion_angles).to(device=self.device, dtype=self.dtype)

        # Reset ligand POT parameters.
        ligand_transform_module.reset_parameters({
            'position_delta': position_delta,
            'orientation_quaternion': orientation_quaternion,
            'torsion_angles': torsion_angles
        })

        # Reset protein parameters (all zeros for native conformation).
        if protein_transform_module is not None:
            protein_transform_module.reset_parameters({
                'protein_torsion_angles': torch.zeros(
                    self.batch_size, protein_transform_module.F, 4,
                    device=self.device, dtype=self.dtype
                )
            })
