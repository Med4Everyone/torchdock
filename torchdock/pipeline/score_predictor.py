"""
XGBoost-based score predictor for early stopping in molecular docking.

This module provides a predictor that loads a pre-trained XGBoost regression
model and estimates final docking convergence scores from gradient tracking
features collected during the early optimization steps.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xgboost as xgb
from scipy import stats


class ScorePredictor:
    """XGBoost-based score predictor for early stopping in molecular docking.

    Loads a pre-trained XGBoost regression model and predicts docking scores
    based on 13-dimensional features extracted from gradient tracking data
    at step 100 (index 9).

    Feature engineering (13 dimensions total):
        - Basic info (2D): heavy_atoms, num_torsions.
        - Static info at step 100 (6D): score_min, score_mean, score_var,
          mean_norm_pos, mean_norm_ori, mean_norm_tor.
        - Evolution dynamics (5D): K_score_global, K_score_late,
          K_norm_pos, K_norm_ori, K_norm_tor.
    """

    # Model path mapping relative to torchdock package root.
    MODEL_PATH_MAP: dict[str, str] = {
        'default': 'pretrain_xgboost/xgboost_model.pkl',
    }

    # Feature names for reference.
    FEATURE_NAMES: list[str] = [
        'heavy_atoms', 'num_torsions',  # Basic info (2D)
        'score_min', 'score_mean', 'score_var', 'mean_norm_pos', 'mean_norm_ori', 'mean_norm_tor',  # Static info (6D)
        'K_score_global', 'K_score_late', 'K_norm_pos', 'K_norm_ori', 'K_norm_tor'  # Evolution dynamics (5D)
    ]

    def __init__(self, config: Any) -> None:
        """Initialize the score predictor with XGBoost model.

        Args:
            config: Configuration object with attributes:
                ``logger`` -- Logger instance.
                ``early_step_model`` -- Model key. Defaults to ``'default'``.
                ``console_output`` -- Whether to output verbose logs.
        """
        self.config = config
        self.logger = config.logger
        self.model: Any | None = None
        self.model_path: str | None = None

        # Determine model path from config.
        self.model_key: str = getattr(config, 'early_step_model', 'default')

        if self.model_key not in self.MODEL_PATH_MAP:
            error_msg = (
                f"Unknown early_step_model: {self.model_key}. "
                f"Available choices: {list(self.MODEL_PATH_MAP.keys())}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Get relative path and convert to absolute path based on torchdock package location.
        relative_path = self.MODEL_PATH_MAP[self.model_key]
        self.model_path = self._get_model_absolute_path(relative_path)

        # Load model if path exists.
        if os.path.exists(self.model_path):
            self._load_model(self.model_path)
        else:
            error_msg = f"Predictor score model path does not exist: {self.model_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    def _get_model_absolute_path(self, relative_path: str) -> str:
        """Convert relative path (from torchdock package root) to absolute path.

        Args:
            relative_path: Path relative to torchdock package root.

        Returns:
            Absolute path to the model file.
        """
        current_file_dir = Path(__file__).parent
        torchdock_root = current_file_dir.parent
        absolute_path = torchdock_root / relative_path
        return str(absolute_path)

    def _load_model(self, model_path: str) -> None:
        """Load pre-trained XGBoost model, auto-migrating pickle to JSON format.

        On first load with a ``.pkl`` file, the model is re-saved as ``.json``
        using XGBoost's native ``save_model`` to avoid cross-version pickle
        warnings. Subsequent loads use the ``.json`` file directly.

        Args:
            model_path: Path to the model file (``.pkl`` format).

        Raises:
            RuntimeError: If the model file cannot be loaded.
        """
        verbose = getattr(self.config, 'console_output', False)
        json_path = Path(model_path).with_suffix('.json')

        # Prefer the migrated JSON file if it already exists.
        if json_path.exists():
            if verbose:
                self.logger.info(f"Loading XGBoost model from JSON: {json_path}")
            self.model = xgb.Booster(model_file=str(json_path))
            return

        # Fall back to pickle, suppress XGBoost version mismatch warnings.
        if verbose:
            self.logger.info(f"Loading predictor score model from: {model_path}")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)

            # Auto-migrate to JSON format for future loads.
            try:
                self.model.save_model(str(json_path))
                if verbose:
                    self.logger.info(f"Migrated XGBoost model to JSON: {json_path}")
            except Exception:
                # Read-only install location; continue with pickle-loaded model.
                pass

            if verbose:
                self.logger.info(f"Successfully loaded XGBoost model from {model_path}")

        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")
            raise

    def is_model_loaded(self) -> bool:
        """Check if the XGBoost model is loaded and ready for prediction.

        Returns:
            True if model is loaded, False otherwise.
        """
        return self.model is not None

    @staticmethod
    def _compute_slope(x: np.ndarray, y: np.ndarray) -> float:
        """Compute slope using linear regression.

        Args:
            x: Independent variable array.
            y: Dependent variable array.

        Returns:
            Slope of the linear regression line.
        """
        if len(x) < 2:
            return 0.0
        slope, _, _, _, _ = stats.linregress(x, y)
        return float(slope)

    def _extract_features(
        self, gradient_tracking: dict[str, list], ligand_info: dict[str, int]
    ) -> np.ndarray:
        """Extract 13-dimensional features from gradient tracking data.

        Uses step 100 (index 9) for static features and computes evolution
        dynamics from all 10 steps (indices 0-9).

        Args:
            gradient_tracking: Dictionary containing gradient and score history:
                ``'position_gradient'`` -- list of ``[batch_size, 3]`` tensors (10 steps).
                ``'orientation_gradient'`` -- list of ``[batch_size, 4]`` tensors (10 steps).
                ``'torsion_gradient'`` -- list of ``[batch_size, T]`` tensors (10 steps).
                ``'torchdock_score'`` -- list of ``[batch_size]`` tensors (10 steps).
            ligand_info: Dictionary containing ligand metadata:
                ``'heavy_atoms'`` -- Number of heavy atoms.
                ``'num_torsions'`` -- Number of torsion angles.

        Returns:
            Feature array of shape ``[13]`` with dtype float32.
        """
        # Get dimensions.
        n_steps = len(gradient_tracking['torchdock_score'])
        step_idx = min(9, n_steps - 1)  # Use step 100 (index 9) or last available step.

        # Convert tensors to numpy arrays.
        pos_grads = [g.cpu().numpy() if isinstance(g, torch.Tensor) else g
                     for g in gradient_tracking['position_gradient']]
        ori_grads = [g.cpu().numpy() if isinstance(g, torch.Tensor) else g
                     for g in gradient_tracking['orientation_gradient']]
        tor_grads = [g.cpu().numpy() if isinstance(g, torch.Tensor) else g
                     for g in gradient_tracking['torsion_gradient']]
        scores = [s.cpu().numpy() if isinstance(s, torch.Tensor) else s
                  for s in gradient_tracking['torchdock_score']]

        n_conformers = scores[0].shape[0]

        # 1. Basic info (2D).
        heavy_atoms = ligand_info.get('heavy_atoms', 0)
        num_torsions = ligand_info.get('num_torsions', 0)

        # 2. Static info at specified step (6D).
        step_scores = scores[step_idx]  # [n_conformers]

        # Select top 10% conformers (elite conformers).
        top_k = max(1, int(n_conformers * 0.1))
        top_indices = np.argsort(step_scores)[:top_k]  # Lowest scores (best).
        elite_scores = step_scores[top_indices]

        score_min = np.min(elite_scores)
        score_mean = np.mean(elite_scores)
        score_var = np.var(elite_scores)

        # Compute gradient norms for elite conformers at specified step.
        pos_grad_step = pos_grads[step_idx][top_indices]  # [top_k, 3]
        ori_grad_step = ori_grads[step_idx][top_indices]  # [top_k, 4]
        tor_grad_step = tor_grads[step_idx][top_indices] if tor_grads[step_idx].size > 0 else np.zeros((top_k, 0))

        # Compute norms.
        pos_norms = np.linalg.norm(pos_grad_step, axis=1)  # [top_k]
        ori_norms = np.linalg.norm(ori_grad_step, axis=1)  # [top_k]

        # For torsion, handle case where num_torsions = 0.
        if tor_grad_step.shape[1] > 0:
            tor_norms = np.linalg.norm(tor_grad_step, axis=1)  # [top_k]
            mean_norm_tor = np.mean(np.log1p(tor_norms))
        else:
            mean_norm_tor = 0.0

        mean_norm_pos = np.mean(np.log1p(pos_norms))
        mean_norm_ori = np.mean(np.log1p(ori_norms))

        # 3. Evolution dynamics (5D).
        # Compute mean elite scores over time.
        elite_means_over_time = []
        for t in range(n_steps):
            t_scores = scores[t]
            t_top_indices = np.argsort(t_scores)[:top_k]
            elite_means_over_time.append(np.mean(t_scores[t_top_indices]))

        elite_means_over_time = np.array(elite_means_over_time)

        # K_score_global: slope from step 10 to 100 (all steps).
        score_offset = np.abs(np.min(elite_means_over_time)) + 1.0
        log_scores = np.log(elite_means_over_time + score_offset)
        steps = np.arange(n_steps)

        k_score_global = self._compute_slope(steps, log_scores)

        # K_score_late: slope from last 5 steps.
        late_start = max(0, n_steps - 5)
        k_score_late = self._compute_slope(steps[late_start:], log_scores[late_start:])

        # Compute gradient norm decay slopes.
        pos_norms_over_time = []
        ori_norms_over_time = []
        tor_norms_over_time = []

        for t in range(n_steps):
            t_pos_grad = pos_grads[t][top_indices]
            t_ori_grad = ori_grads[t][top_indices]
            t_tor_grad = tor_grads[t][top_indices] if tor_grads[t].size > 0 else np.zeros((top_k, 0))

            t_pos_norms = np.linalg.norm(t_pos_grad, axis=1)
            t_ori_norms = np.linalg.norm(t_ori_grad, axis=1)

            pos_norms_over_time.append(np.mean(t_pos_norms))
            ori_norms_over_time.append(np.mean(t_ori_norms))

            if t_tor_grad.shape[1] > 0:
                t_tor_norms = np.linalg.norm(t_tor_grad, axis=1)
                tor_norms_over_time.append(np.mean(t_tor_norms))
            else:
                tor_norms_over_time.append(0.0)

        pos_norms_over_time = np.array(pos_norms_over_time)
        ori_norms_over_time = np.array(ori_norms_over_time)
        tor_norms_over_time = np.array(tor_norms_over_time)

        # Compute slopes on log scale.
        log_pos_norms = np.log1p(pos_norms_over_time)
        log_ori_norms = np.log1p(ori_norms_over_time)
        log_tor_norms = np.log1p(tor_norms_over_time)

        k_norm_pos = self._compute_slope(steps, log_pos_norms)
        k_norm_ori = self._compute_slope(steps, log_ori_norms)
        k_norm_tor = self._compute_slope(steps, log_tor_norms)

        # Combine all features.
        features = np.array([
            heavy_atoms, num_torsions,  # Basic info (2D)
            score_min, score_mean, score_var, mean_norm_pos, mean_norm_ori, mean_norm_tor,  # Static info (6D)
            k_score_global, k_score_late, k_norm_pos, k_norm_ori, k_norm_tor  # Evolution dynamics (5D)
        ], dtype=np.float32)

        return features

    def predict(
        self, gradient_tracking: dict[str, list], ligand_info: dict[str, int]
    ) -> float:
        """Predict docking score based on gradient tracking data.

        Extracts 13-dimensional features from gradient tracking data and
        uses XGBoost to predict the final convergence score.

        Args:
            gradient_tracking: Dictionary containing gradient and score data
                collected at steps 10-100 (10 collection points, indices 0-9).
                Expected keys:
                ``'position_gradient'`` -- list of ``[batch_size, 3]`` tensors.
                ``'orientation_gradient'`` -- list of ``[batch_size, 4]`` tensors.
                ``'torsion_gradient'`` -- list of ``[batch_size, T]`` tensors.
                ``'torchdock_score'`` -- list of ``[batch_size]`` tensors.
            ligand_info: Dictionary containing ligand metadata.
                Expected keys:
                ``'heavy_atoms'`` -- Number of heavy atoms.
                ``'num_torsions'`` -- Number of torsion angles.

        Returns:
            Predicted docking score.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Cannot make predictions.")

        # Extract 13-dimensional features.
        features = self._extract_features(gradient_tracking, ligand_info)  # [13]

        # Reshape to [1, 13] for model input.
        features_2d = features.reshape(1, -1)

        # Use XGBoost model to predict score.
        dmatrix = xgb.DMatrix(features_2d)
        prediction = self.model.predict(dmatrix)[0]

        return float(prediction)
