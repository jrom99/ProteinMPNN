"""utils.py - ProteinMPNN Model utilities.

DESCRIPTION
    This module contains utility functions for loading model checkpoints.

FUNCTIONS
    get_checkpoint(model_name, ca_only, use_soluble_model)
        Return the path to a model checkpoint file.
"""

import logging
from importlib.resources import files
from pathlib import Path

import protein_mpnn

LOGGER = logging.getLogger(__name__)


class CheckpointNotFoundError(FileNotFoundError):
    """If a pytorch checkpoint file is missing."""


def get_checkpoint(model_name: str, *, ca_only: bool, use_soluble_model: bool) -> Path:
    """Return model checkpoint file."""
    if Path(model_name).is_file():
        checkpoint_path = Path(model_name)

    weights = files(protein_mpnn) / "weights"

    if not weights.is_dir():
        msg = f"Missing WEIGHTS directory: {weights!s}"
        raise CheckpointNotFoundError(msg)

    if ca_only:
        LOGGER.info("Using CA-ProteinMPNN!")
        checkpoint_path = weights / "ca_model_weights" / f"{model_name}.pt"
        if use_soluble_model:
            msg = "CA-SolubleMPNN is not available yet"
            raise ValueError(msg)
    elif use_soluble_model:
        checkpoint_path = weights / "soluble_model_weights" / f"{model_name}.pt"
    else:
        checkpoint_path = weights / "vanilla_model_weights" / f"{model_name}.pt"

    if not checkpoint_path.is_file():
        raise CheckpointNotFoundError(checkpoint_path)

    return Path(str(checkpoint_path))
