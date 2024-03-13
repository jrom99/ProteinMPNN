import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from protein_mpnn.data_processing.utils import parse_fasta
from protein_mpnn.features.build_features import S_to_seq
from protein_mpnn.features.build_features import scores as _scores
from protein_mpnn.models.inference_model import ProteinMPNN

LOGGER = logging.getLogger(__name__)


def calculate_score_only(
    path_to_fasta: str | None,
    num_batches: int,
    alphabet_dict: dict[str, int],
    device: torch.device,
    model: ProteinMPNN,
    out_folder: Path,
    X: Tensor,
    S: Tensor,
    mask: Tensor,
    chain_M: Tensor,
    chain_encoding_all: Tensor,
    chain_M_pos: Tensor,
    residue_idx: Tensor,
    name_: str,
):
    SCORE_ONLY_FOLDER = out_folder / "score_only"
    SCORE_ONLY_FOLDER.mkdir(exist_ok=True)

    loop_c = 0
    if path_to_fasta:
        _, fasta_seqs = parse_fasta(path_to_fasta, omit="/")
        loop_c = len(fasta_seqs)

    for fc in range(1 + loop_c):
        if fc == 0:
            structure_sequence_score_file = SCORE_ONLY_FOLDER / f"{name_}_pdb"
        else:
            structure_sequence_score_file = SCORE_ONLY_FOLDER / f"{name_}_fasta_{fc}"
        native_score_list = []
        global_native_score_list = []
        if fc > 0:
            input_seq_length = len(fasta_seqs[fc - 1])
            S_input = torch.tensor(
                [alphabet_dict[AA] for AA in fasta_seqs[fc - 1]],
                device=device,
            )[None, :].repeat(X.shape[0], 1)
            S[:, :input_seq_length] = (
                S_input  # assumes that S and S_input are alphabetically sorted for masked_chains
            )
        for _ in range(num_batches):
            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = model(
                X,
                S,
                mask,
                chain_M * chain_M_pos,
                residue_idx,
                chain_encoding_all,
                randn_1,
            )
            mask_for_loss = mask * chain_M * chain_M_pos
            scores = _scores(S, log_probs, mask_for_loss)
            native_score = scores.cpu().data.numpy()
            native_score_list.append(native_score)
            global_scores = _scores(S, log_probs, mask)
            global_native_score = global_scores.cpu().data.numpy()
            global_native_score_list.append(global_native_score)
        native_score = np.concatenate(native_score_list, 0)
        global_native_score = np.concatenate(global_native_score_list, 0)
        ns_mean = native_score.mean()
        ns_mean_print = np.format_float_positional(np.float32(ns_mean), unique=False, precision=4)
        ns_std = native_score.std()
        ns_std_print = np.format_float_positional(np.float32(ns_std), unique=False, precision=4)

        global_ns_mean = global_native_score.mean()
        global_ns_mean_print = np.format_float_positional(
            np.float32(global_ns_mean), unique=False, precision=4
        )
        global_ns_std = global_native_score.std()
        global_ns_std_print = np.format_float_positional(
            np.float32(global_ns_std), unique=False, precision=4
        )

        ns_sample_size = native_score.shape[0]
        seq_str = S_to_seq(S[0,], chain_M[0,])
        np.savez(
            structure_sequence_score_file,
            score=native_score,
            global_score=global_native_score,
            S=S[0,].cpu().numpy(),
            seq_str=seq_str,
        )

        _a = name_ if fc == 0 else f"{name_}_{fc}"
        _b = "PDB" if fc == 0 else "FASTA"
        _data = {
            "mean": ns_mean_print,
            "std": ns_std_print,
            "sample size": ns_sample_size,
            "global mean": global_ns_mean_print,
            "global std": global_ns_std_print,
            "global sample size": ns_sample_size
        }
        _data_print = ",".join(f"{k}: {v}" for k, v in _data.items())

        LOGGER.debug(f"Score for {_a} from {_b}: {_data_print}")

