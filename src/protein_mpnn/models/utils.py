import logging
from pathlib import Path
from importlib.resources import files

import protein_mpnn


LOGGER = logging.getLogger(__name__)
WEIGHTS_DIR = Path(__file__).resolve()


def get_model(model_name: str, ca_only: bool, use_soluble_model: bool) -> Path:
    weights = files(protein_mpnn) / "weights"
    LOGGER.debug(weights)

    if weights.is_dir():
        LOGGER.debug("Weights were found")

    if Path(model_name).is_file():
        checkpoint_path = Path(model_name)
    elif ca_only:
        LOGGER.info("Using CA-ProteinMPNN!")
        files("protein_mpnn")
        checkpoint_path = WEIGHTS_DIR / "ca_model_weights" / f"{model_name}.pt"
        if use_soluble_model:
            raise ValueError("CA-SolubleMPNN is not available yet")
    elif use_soluble_model:
        checkpoint_path = WEIGHTS_DIR / "soluble_model_weights" / f"{model_name}.pt"
    else:
        checkpoint_path = WEIGHTS_DIR / "vanilla_model_weights" / f"{model_name}.pt"

    return checkpoint_path


if __name__ == "__main__":
    log_fmt = "{asctime} - {name} - {levelname} - {message}"
    logging.basicConfig(level=logging.DEBUG, format=log_fmt, style="{")
    ckp = get_model("v022", ca_only=False, use_soluble_model=False)
    LOGGER.debug(f"Using model {ckp}")
