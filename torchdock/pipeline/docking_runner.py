"""
Molecular docking runner and CLI entry point.

This module provides the main docking workflow: parsing configuration,
loading molecular data, running conformer search, and saving results.
It supports Vina and Vinardo scoring functions.
"""

# Copyright (c) 2026 The TorchDock Authors
#
# This file is part of TorchDock.
# Licensed under the Apache License, Version 2.0. See LICENSE file for details.

import argparse
import json
import logging
import os
import time
from typing import Any

from torchdock.config.config import Config
from torchdock.utils.logging import setup_logger


def set_num_threads_and_device(config: Any, num_threads: int, device_arg: str) -> None:
    """Set CPU thread limits and device for PyTorch.

    Configures environment variables (for BLAS/LAPACK backends), PyTorch
    runtime thread pools, and the compute device.

    **Important**: Must be called BEFORE importing torch or numpy elsewhere.

    Args:
        config: Configuration object with a ``setattr`` method.
        num_threads: Number of CPU threads to use.
        device_arg: Device string (e.g., ``'cpu'``, ``'cuda:0'``).
    """
    # --- 1. Set environment variables for BLAS libraries ---
    env_vars = {
        "OMP_NUM_THREADS": str(num_threads),
        "OPENBLAS_NUM_THREADS": str(num_threads),
        "MKL_NUM_THREADS": str(num_threads),
        "VECLIB_MAXIMUM_THREADS": str(num_threads),
        "NUMEXPR_NUM_THREADS": str(num_threads),
        "CV_NUM_THREADS": str(num_threads),
    }
    for k, v in env_vars.items():
        os.environ[k] = v

    # --- 2. Configure PyTorch thread pools ---
    import torch
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already set in parent process or after parallel work started.
        pass

    # --- 3. Set device ---
    if device_arg.startswith("cuda"):
        if torch.cuda.is_available():
            config.setattr("device", device_arg)
            if ":" in device_arg:
                device_id = int(device_arg.split(":")[1])
                torch.cuda.set_device(device_id)
        else:
            config.logger.warning("CUDA not available, falling back to CPU.")
            config.setattr("device", "cpu")
    else:
        config.setattr("device", "cpu")


def parse_box_file(box_file_path: str) -> dict[str, list[float]]:
    """Parse a box configuration file in JSON format.

    Expected JSON structure::

        {"center": [x, y, z], "size": [dx, dy, dz]}

    Args:
        box_file_path: Path to the box JSON file.

    Returns:
        Dictionary with ``'center'`` and ``'size'`` keys, each mapping
        to a list of 3 floats.

    Raises:
        FileNotFoundError: If the box file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If required keys are missing or values are invalid.
    """
    if not os.path.isfile(box_file_path):
        raise FileNotFoundError(f"Box file not found: {box_file_path}")

    with open(box_file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in box file {box_file_path}: {e.msg}", e.doc, e.pos
            )

    if 'center' not in data:
        raise ValueError(f"Missing required key 'center' in box file: {box_file_path}")
    if 'size' not in data:
        raise ValueError(f"Missing required key 'size' in box file: {box_file_path}")

    center = data['center']
    size = data['size']

    if not isinstance(center, (list, tuple)) or len(center) != 3:
        raise ValueError(f"'center' must be a list/tuple of 3 numbers in {box_file_path}")
    try:
        center = [float(x) for x in center]
    except (TypeError, ValueError):
        raise ValueError(f"'center' contains non-numeric values in {box_file_path}")

    if not isinstance(size, (list, tuple)) or len(size) != 3:
        raise ValueError(f"'size' must be a list/tuple of 3 numbers in {box_file_path}")
    try:
        size = [float(x) for x in size]
    except (TypeError, ValueError):
        raise ValueError(f"'size' contains non-numeric values in {box_file_path}")

    if any(s <= 0 for s in size):
        raise ValueError(f"'size' values must be positive in {box_file_path}")

    return {"center": center, "size": size}


def save_early_stop_result(output_path: str, predicted_score: float) -> None:
    """Save a minimal PDBQT result file for early stop mode.

    The file contains only a MODEL header and REMARK line with the
    predicted score, following Vina format conventions.

    Args:
        output_path: Path to output PDBQT file.
        predicted_score: The predicted convergence score.
    """
    if hasattr(predicted_score, 'item'):
        predicted_score = predicted_score.item()

    content_lines = [
        "MODEL 1\n",
        f"REMARK VINA RESULT: {predicted_score:>12.3f} "
        f"{0.000:>12.3f} {0.000:>12.3f}\n",
        "ENDMDL\n",
    ]

    with open(output_path, 'w') as f:
        f.writelines(content_lines)


def save_gradient_tracking(
    config: Any,
    conformer_search: Any,
    result: list[float],
    total_cpu_time: float | None = None,
) -> None:
    """Save epoch0 gradient tracking data and docking results to ``.pth`` file.

    Args:
        config: Configuration object.
        conformer_search: SMACConformerSearch instance with gradient data.
        result: Score list ``[torchdock, total, inter, intra, unbound]``.
        total_cpu_time: Total CPU time elapsed in seconds. Defaults to None.
    """
    import numpy as np
    import torch

    epoch0_gradient_tracking = conformer_search.epoch0_gradient_tracking

    torchdock_score = result[0]
    total_score = result[1]
    inter_score = result[2]
    intra_score = result[3]
    unbound_score = result[4]

    # Convert gradient tracking data from list of tensors to stacked numpy arrays.
    gradient_data: dict[str, Any] = {}

    if len(epoch0_gradient_tracking['position_gradient']) > 0:
        gradient_data['position_gradient'] = torch.stack(epoch0_gradient_tracking['position_gradient']).cpu().numpy()
        gradient_data['orientation_gradient'] = torch.stack(epoch0_gradient_tracking['orientation_gradient']).cpu().numpy()
        gradient_data['torsion_gradient'] = torch.stack(epoch0_gradient_tracking['torsion_gradient']).cpu().numpy()
    else:
        config.logger.warning("No gradient data collected in epoch0.")
        gradient_data['position_gradient'] = np.array([])
        gradient_data['orientation_gradient'] = np.array([])
        gradient_data['torsion_gradient'] = np.array([])

    if len(epoch0_gradient_tracking['torchdock_score']) > 0:
        gradient_data['torchdock_score'] = torch.stack(epoch0_gradient_tracking['torchdock_score']).cpu().numpy()
        gradient_data['total_score'] = torch.stack(epoch0_gradient_tracking['total_score']).cpu().numpy()
        gradient_data['inter_score'] = torch.stack(epoch0_gradient_tracking['inter_score']).cpu().numpy()
        gradient_data['ligand_intra_score'] = torch.stack(epoch0_gradient_tracking['ligand_intra_score']).cpu().numpy()
        gradient_data['protein_intra_score'] = torch.stack(epoch0_gradient_tracking['protein_intra_score']).cpu().numpy()
    else:
        gradient_data['torchdock_score'] = np.array([])
        gradient_data['total_score'] = np.array([])
        gradient_data['inter_score'] = np.array([])
        gradient_data['ligand_intra_score'] = np.array([])
        gradient_data['protein_intra_score'] = np.array([])

    # Save CPU time data (list of floats, not tensors).
    if len(epoch0_gradient_tracking['cpu_time']) > 0:
        gradient_data['cpu_time'] = np.array(epoch0_gradient_tracking['cpu_time'])
    else:
        gradient_data['cpu_time'] = np.array([])

    docking_result = {
        'top1_torchdock_score': torchdock_score,
        'top1_total_score': total_score,
        'top1_inter_score': inter_score,
        'top1_intra_score': intra_score,
        'top1_unbound_score': unbound_score,
    }

    # Extract ligand basic information.
    ligand_loader = conformer_search.ligand_loader
    total_atoms = ligand_loader.atoms_num
    num_hydrogens = len(ligand_loader.h_atom_indices)
    heavy_atoms = total_atoms - num_hydrogens
    num_torsions = len(ligand_loader.torsions)

    ligand_info = {
        'heavy_atoms': heavy_atoms,
        'num_torsions': num_torsions,
    }

    save_data = {
        'gradient_tracking': gradient_data,
        'docking_result': docking_result,
        'ligand_info': ligand_info,
        'total_cpu_time': total_cpu_time if total_cpu_time is not None else 0.0,
    }

    # Determine save path (same as output_path but with .pth extension).
    output_path = config.output_file_path
    save_path = os.path.splitext(output_path)[0] + '.pth'

    torch.save(save_data, save_path)
    config.logger.info(f"Gradient tracking data saved to: {save_path}")


def run_docking(config: Any) -> list[float]:
    """Run molecular docking based on the provided configuration.

    Args:
        config: Configuration object with all docking parameters.

    Returns:
        Score list: ``[torchdock_score, total_score, inter_score,
        intra_score, unbound_score]``.
    """
    cpu_times_start = os.times()
    wall_time_start = time.time()

    import random as pyrandom

    import numpy as np
    import torch

    from torchdock.data.vina_dataloader import VinaProteinLoader, VinaLigandLoader
    from torchdock.model.conformer_transformer import LigandConformerTransform, ProteinConformerTransform
    from torchdock.model.vina_scoring import VinaScoreModel
    from torchdock.pipeline.conformation_search import SMACConformerSearch
    from torchdock.pipeline.postprocessor import VinaResultFormatProcessor

    logger = config.logger

    # Set random seeds for reproducibility.
    pyrandom.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device != "cpu":
        torch.cuda.manual_seed_all(config.seed)

    # Load ligand and protein data loaders.
    if config.score_function in ["vina", "vinardo"]:
        ligand_loader = VinaLigandLoader(config)
        protein_loader = VinaProteinLoader(config)
    else:
        raise ValueError(f"Invalid scoring function: {config.score_function}")

    logger.info(ligand_loader)
    protein_loader.partition_pocket(flex_dock=config.flex_dock, flexible_residues=config.flex_residues)
    logger.info(protein_loader)

    # Count rotatable torsions from ligand and flexible protein residues.
    ligand_torsion_num = min(config.ligand_max_torsions, len(ligand_loader.torsions))
    protein_torsion_num = np.sum(~np.all(protein_loader.flex_torsions == -1, axis=-1))
    total_torsion_num = ligand_torsion_num + protein_torsion_num
    logger.info(f"Total valid torsions: {total_torsion_num}")

    # Initialize scoring model and prepare complex.
    if config.score_function in ["vina", "vinardo"]:
        scoring_model = VinaScoreModel(config)
    else:
        raise ValueError(f"Invalid scoring function: {config.score_function}")
    scoring_model.prepare_complex(ligand_loader, protein_loader)

    # Score only mode.
    if config.score_only:
        device = torch.device(config.device)
        dtype = getattr(torch, config.dtype)

        ligand_coords = torch.from_numpy(ligand_loader.ligand_coords).unsqueeze(0).to(device=device, dtype=dtype)
        flex_coords = torch.from_numpy(protein_loader.flex_coords).unsqueeze(0).to(device=device, dtype=dtype)

        score = scoring_model(ligand_coords, flex_coords, include_pocket_penalty=True)
        total_score_t, inter_score_t, ligand_intra_t, protein_intra_t = score

        total_score = total_score_t.item()
        inter_score = inter_score_t.item()
        intra_score = ligand_intra_t.item() + protein_intra_t.item()
        unbound_score = intra_score

        weight_torsion = scoring_model.weight_torsion.item()
        degrees_of_freedom = ligand_loader.degrees_of_freedom

        if config.score_function in ["vina", "vinardo"]:
            torchdock_score = (total_score - unbound_score) / (1.0 + weight_torsion * degrees_of_freedom)
        else:
            torchdock_score = 0.0

        result = [torchdock_score, total_score, inter_score, intra_score, unbound_score]

        cpu_times_end = os.times()
        wall_time_end = time.time()
        cpu_time_used = (
            cpu_times_end.user - cpu_times_start.user
            + cpu_times_end.system - cpu_times_start.system
        )
        wall_time_used = wall_time_end - wall_time_start

        logger.info(f"TorchDock Score: {torchdock_score}")
        logger.info(f"Total Score: {total_score}")
        logger.info(f"Inter Score: {inter_score}")
        logger.info(f"Intra Score: {intra_score}")
        logger.info(f"CPU Time: {cpu_time_used:.2f}s")
        logger.info(f"Wall Time: {wall_time_used:.2f}s")

        return result

    # Initialize conformer transformers.
    ligand_transformer = LigandConformerTransform(config, ligand_loader)
    protein_transformer = ProteinConformerTransform(config, protein_loader)

    # Initialize and perform conformer search.
    conformer_search = SMACConformerSearch(
        config, ligand_loader, protein_loader, scoring_model,
        ligand_transformer, protein_transformer,
    )
    selected_poses = conformer_search.search()

    # Early stop mode.
    if config.early_stop:
        predicted_score = selected_poses

        save_early_stop_result(config.output_file_path, predicted_score)

        cpu_times_end = os.times()
        wall_time_end = time.time()
        cpu_time_used = (
            cpu_times_end.user - cpu_times_start.user
            + cpu_times_end.system - cpu_times_start.system
        )
        wall_time_used = wall_time_end - wall_time_start

        result = [predicted_score, predicted_score, predicted_score, 0.0, 0.0]

        if config.save_gradient:
            save_gradient_tracking(config, conformer_search, result, cpu_time_used)

        logger.info(f"Predicted convergence score: {predicted_score}")
        logger.info(f"CPU Time: {cpu_time_used:.2f}s")
        logger.info(f"Wall Time: {wall_time_used:.2f}s")
        logger.info(f"Early stop result saved to {config.output_file_path}")

        return result

    # Process and save results in Vina result format.
    vina_result_processor = VinaResultFormatProcessor(config, ligand_loader, protein_loader, scoring_model)
    result = vina_result_processor.process_and_save_results(selected_poses, config.output_file_path)

    cpu_times_end = os.times()
    wall_time_end = time.time()
    cpu_time_used = (
        cpu_times_end.user - cpu_times_start.user
        + cpu_times_end.system - cpu_times_start.system
    )
    wall_time_used = wall_time_end - wall_time_start

    if config.save_gradient:
        save_gradient_tracking(config, conformer_search, result, cpu_time_used)

    logger.info(f"Top-1 TorchDock Score: {result[0]}")
    logger.info(f"Top-1 Total Score: {result[1]}")
    logger.info(f"Top-1 Inter Score: {result[2]}")
    logger.info(f"Top-1 Intra Score: {result[3]}")
    logger.info(f"CPU Time: {cpu_time_used:.2f}s")
    logger.info(f"Wall Time: {wall_time_used:.2f}s")
    logger.info(f"Docking result saved to {config.output_file_path}")

    return result


def docking(**kwargs: Any) -> list[float]:
    """Execute docking based on the provided keyword arguments.

    Args:
        **kwargs: Docking parameters including ligand/protein paths,
            box configuration, and optional flags.

    Returns:
        Score list: ``[torchdock_score, total_score, inter_score,
        intra_score, unbound_score]``.
    """
    config = Config(kwargs.get('config_file_path', None))

    # Set random seed for reproducibility.
    seed = int.from_bytes(os.urandom(4), byteorder='big') if config.seed is None else int(config.seed)
    config.setattr("seed", seed)

    # Set basic parameters.
    config.setattr("ligand_file_path", kwargs.get('ligand_pdbqt_path', None))
    config.setattr("protein_file_path", kwargs.get('protein_pdbqt_path', None))
    config.setattr("output_file_path", kwargs.get('output_path', None))
    config.setattr("log_file_path", kwargs.get('log_path', None))
    config.setattr("num_threads", kwargs.get('num_workers', config.num_threads))

    if 'score_only' in kwargs and kwargs['score_only'] is not None:
        config.setattr("score_only", kwargs['score_only'])
    if 'early_stop' in kwargs and kwargs['early_stop'] is not None:
        config.setattr("early_stop", kwargs['early_stop'])
    if 'save_gradient' in kwargs and kwargs['save_gradient'] is not None:
        config.setattr("save_gradient", kwargs['save_gradient'])

    # Handle box parameters.
    box_file_path = kwargs.get('box_file_path', None)
    box_center = kwargs.get('box_center', None)
    box_size = kwargs.get('box_size', None)

    if box_file_path is not None:
        box_info = parse_box_file(box_file_path)
        config.setattr("box_center", box_info["center"])
        config.setattr("box_size", box_info["size"])
    elif box_center is not None and box_size is not None:
        config.setattr("box_center", [float(x) for x in box_center])
        config.setattr("box_size", [float(x) for x in box_size])
    else:
        raise ValueError(
            "Either --box_file_path must be provided, "
            "or both --box_center and --box_size must be specified."
        )

    # Set flex docking parameters.
    if 'flex' in kwargs and kwargs['flex'] is not None:
        config.setattr("flex_dock", kwargs['flex'])
    config.setattr("flex_residues", kwargs.get('flex_residues', None))

    # Initialize logger.
    if 'verbose' in kwargs and kwargs['verbose'] is not None:
        config.setattr("console_output", kwargs['verbose'])

    logger = setup_logger(
        name='docking', level=config.log_level,
        log_file=config.log_file_path, console_output=config.console_output,
    )
    config.setattr("logger", logger)

    # Validate input files.
    if not os.path.isfile(config.ligand_file_path):
        config.logger.error(f"Ligand file not found: {config.ligand_file_path}")
        raise FileNotFoundError(f"Ligand file not found: {config.ligand_file_path}")
    if not os.path.isfile(config.protein_file_path):
        config.logger.error(f"Protein file not found: {config.protein_file_path}")
        raise FileNotFoundError(f"Protein file not found: {config.protein_file_path}")

    # Create output directory.
    output_dir = os.path.dirname(config.output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Set number of CPU threads and device.
    set_num_threads_and_device(config, config.num_threads, kwargs.get('device', config.device))

    config.logger.info(f"Config: {config}")

    result = run_docking(config)

    # Ensure log file is fully written to disk before returning.
    if config.log_file_path:
        for handler in config.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
                try:
                    os.fsync(handler.stream.fileno())
                except (AttributeError, OSError):
                    pass

    return result


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for TorchDock.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="TorchDock docking parameters configuration")

    # Basic parameters.
    parser.add_argument("-l", "--ligand_pdbqt_path", type=str, required=True, help="Ligand PDBQT file path.")
    parser.add_argument("-r", "--protein_pdbqt_path", type=str, required=True, help="Protein PDBQT file path.")
    parser.add_argument("-b", "--box_file_path", type=str, default=None, help="Box configuration JSON file path.")
    parser.add_argument("-bc", "--box_center", type=float, nargs=3, default=None, help="Box center (x y z).")
    parser.add_argument("-bs", "--box_size", type=float, nargs=3, default=None, help="Box size (dx dy dz).")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Output PDBQT file path.")
    parser.add_argument("-log", "--log_path", type=str, default=None, help="Log file path.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode.")
    parser.add_argument("-nw", "--num_workers", type=int, default=None, help="Number of CPU workers.")

    # Flex docking parameters.
    parser.add_argument("-f", "--flex", action="store_true", help="Enable flexible docking.")
    parser.add_argument(
        "--flex_residues", type=str, default=None,
        help="Flexible residues (e.g., 'A:123,A:125,B:45'). Auto-detected if not provided.",
    )

    # Scoring and control flags.
    parser.add_argument("-sc", "--score_only", action="store_true", help="Only perform scoring (no search).")
    parser.add_argument("-d", "--device", type=str, default=None, help="Compute device (e.g., 'cpu', 'cuda:0').")
    parser.add_argument("-c", "--config_file_path", type=str, default=None, help="Configuration file path.")

    # Early stopping parameters.
    parser.add_argument("-es", "--early_stop", action="store_true", help="Enable early stopping mode.")
    parser.add_argument("-sg", "--save_gradient", action="store_true", help="Save gradient tracking data.")

    return parser


def main(args: argparse.Namespace | None = None) -> list[float]:
    """Parse command-line arguments and execute docking.

    Args:
        args: Pre-parsed arguments. If None, reads from ``sys.argv``.

    Returns:
        Score list from :func:`docking`.
    """
    parser = create_parser()
    if args is None:
        args = parser.parse_args()

    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    return docking(**kwargs)


if __name__ == "__main__":
    main()