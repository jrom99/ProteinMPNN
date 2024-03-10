from pathlib import Path


WEIGHTS_DIR = Path(__file__).resolve()


def get_model(model_name: str, ca_only: bool, use_soluble_model: bool) -> Path:
    if Path(model_name).is_file():
        checkpoint_path = Path(model_name)
    elif ca_only:
        # LOGGER.info("Using CA-ProteinMPNN!")
        checkpoint_path = WEIGHTS_DIR / "ca_model_weights" / f"{model_name}.pt"
        if use_soluble_model:
            raise ValueError("CA-SolubleMPNN is not available yet")
    elif use_soluble_model:
        checkpoint_path = WEIGHTS_DIR / "soluble_model_weights" / f"{model_name}.pt"
    else:
        checkpoint_path = WEIGHTS_DIR / "vanilla_model_weights" / f"{model_name}.pt"
    
    return checkpoint_path


# TODO: maybe put each on its own file?

def calculate_conditional_probs(*args, **kwargs) -> None: ...

def calculate_unconditional_probs(*args, **kwargs) -> None: ...

def calculate_score_only(*args, **kwargs) -> None: ...

def generate_sequences(*args, **kwargs) -> None: ...
