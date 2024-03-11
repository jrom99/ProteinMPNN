import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from protein_mpnn.models.inference_model import ProteinMPNN

LOGGER = logging.getLogger(__name__)


def calculate_conditional_probs(
    conditional_probs_only_backbone: bool,
    num_batches: int,
    model: ProteinMPNN,
    output_folder: Path,
    X: Tensor,
    S: Tensor,
    mask: Tensor,
    chain_M: Tensor,
    chain_encoding_all: Tensor,
    chain_M_pos: Tensor,
    residue_idx: Tensor,
    name_: str,
):
    LOGGER.info(f"Calculating conditional probabilities for {name_}")

    COND_PROBS_ONLY_FOLDER = output_folder / "conditional_probs_only"
    COND_PROBS_ONLY_FOLDER.mkdir(exist_ok=True)

    conditional_probs_only_file = COND_PROBS_ONLY_FOLDER / name_
    log_conditional_probs_list = []
    for _ in range(num_batches):
        randn_1 = torch.randn(chain_M.shape, device=X.device)
        log_conditional_probs = model.conditional_probs(
            X,
            S,
            mask,
            chain_M * chain_M_pos,
            residue_idx,
            chain_encoding_all,
            randn_1,
            conditional_probs_only_backbone,
        )
        log_conditional_probs_list.append(log_conditional_probs.cpu().numpy())
    concat_log_p = np.concatenate(log_conditional_probs_list, 0)  # [B, L, 21]
    mask_out = (chain_M * chain_M_pos * mask)[0,].cpu().numpy()
    np.savez(
        conditional_probs_only_file,
        log_p=concat_log_p,
        S=S[0,].cpu().numpy(),
        mask=mask[0,].cpu().numpy(),
        design_mask=mask_out,
    )


def calculate_unconditional_probs(
    num_batches: int,
    model: ProteinMPNN,
    output_folder: Path,
    X: Tensor,
    S: Tensor,
    mask: Tensor,
    chain_M: Tensor,
    chain_encoding_all: Tensor,
    chain_M_pos: Tensor,
    residue_idx: Tensor,
    name_: str,
):
    LOGGER.info(f"Calculating sequence unconditional probabilities for {name_}")

    UNCOND_PROBS_ONLY_FOLDER = output_folder / "unconditional_probs_only"
    UNCOND_PROBS_ONLY_FOLDER.mkdir(exist_ok=True)

    unconditional_probs_only_file = UNCOND_PROBS_ONLY_FOLDER / name_
    log_unconditional_probs_list = []
    for _ in range(num_batches):
        log_unconditional_probs = model.unconditional_probs(
            X, mask, residue_idx, chain_encoding_all
        )
        log_unconditional_probs_list.append(log_unconditional_probs.cpu().numpy())
    concat_log_p = np.concatenate(log_unconditional_probs_list, 0)  # [B, L, 21]
    mask_out = (chain_M * chain_M_pos * mask)[0,].cpu().numpy()
    np.savez(
        unconditional_probs_only_file,
        log_p=concat_log_p,
        S=S[0,].cpu().numpy(),
        mask=mask[0,].cpu().numpy(),
        design_mask=mask_out,
    )
