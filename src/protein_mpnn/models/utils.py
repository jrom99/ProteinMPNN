import logging
from pathlib import Path
from importlib.resources import files

import protein_mpnn


LOGGER = logging.getLogger(__name__)


class CheckpointNotFoundError(FileNotFoundError):
    """If a pytorch checkpoint file is missing."""

class CheckpointDirNotFoundError(CheckpointNotFoundError):
    """If the weights dir is missing."""
    def __init__(self, file: str | Path) -> None:
        super().__init__(f"Missing WEIGHTS directory: {file!s}")


def get_model(model_name: str, *, ca_only: bool, use_soluble_model: bool) -> Path:
    """Return model checkpoint file."""
    if Path(model_name).is_file():
        checkpoint_path = Path(model_name)

    weights = files(protein_mpnn) / "weights"
    LOGGER.debug(f"WEIGHTS DIR: {weights!r}")

    if not weights.is_dir():
        raise CheckpointDirNotFoundError(str(weights))

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
