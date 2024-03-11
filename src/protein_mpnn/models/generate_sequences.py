import logging
import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from protein_mpnn.features.build_features import scores as _scores, S_to_seq

from protein_mpnn.models.inference_model import ProteinMPNN
from protein_mpnn import __version__


LOGGER = logging.getLogger(__name__)


def generate_sequences(
    pssm_multi: float,
    pssm_log_odds_flag: bool,
    pssm_bias_flag: bool,
    model_name: str,
    ca_only: bool,
    save_score: bool,
    save_probs: bool,
    seed: int,
    num_batches: int,
    batch_copies: int,
    temperatures: list[float],
    omit_AAs_np: NDArray[np.float32],
    tied_positions_dict,
    bias_AAs_np,
    model: ProteinMPNN,
    output_folder: Path,
    score_list: list,
    global_score_list: list,
    all_probs_list: list,
    all_log_probs_list: list,
    S_sample_list: list,
    chain_list_list: list,
    visible_list_list: list,
    masked_list_list: list,
    masked_chain_length_list_list: list,
    tied_pos_list_of_lists_list: list,
    X: Tensor,
    S: Tensor,
    mask: Tensor,
    chain_M: Tensor,
    chain_encoding_all: Tensor,
    chain_M_pos: Tensor,
    omit_AA_mask: Tensor,
    residue_idx: Tensor,
    pssm_coef: Tensor,
    pssm_bias: Tensor,
    bias_by_res_all: Tensor,
    tied_beta: Tensor,
    pssm_log_odds_mask: Tensor,
    name_: str,
):
    LOGGER.info(f"Generating sequences for: {name_}")

    SEQS_FOLDER = output_folder / "seqs"
    SCORES_FOLDER = output_folder / "scores"
    PROBS_FOLDER = output_folder / "probs"

    SEQS_FOLDER.mkdir(exist_ok=True)

    if save_score:
        SCORES_FOLDER.mkdir(exist_ok=True)

    if save_probs:
        PROBS_FOLDER.mkdir(exist_ok=True)

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
    scores = _scores(S, log_probs, mask_for_loss)  # score only the redesigned part
    native_score = scores.cpu().data.numpy()
    global_scores = _scores(S, log_probs, mask)  # score the whole structure-sequence
    global_native_score = global_scores.cpu().data.numpy()

    # Generate some sequences
    ali_file = SEQS_FOLDER / f"{name_}.fa"
    score_file = SCORES_FOLDER / f"{name_}.npz"
    probs_file = PROBS_FOLDER / f"{name_}.npz"

    t0 = time.perf_counter()
    with open(ali_file, "w") as f:
        for temp in temperatures:
            for j in range(num_batches):
                randn_2 = torch.randn(chain_M.shape, device=X.device)
                if tied_positions_dict is None:
                    sample_dict = model.sample(
                        X,
                        randn_2,
                        S,
                        chain_M,
                        chain_encoding_all,
                        residue_idx,
                        mask=mask,
                        temperature=temp,
                        omit_AAs_np=omit_AAs_np,
                        bias_AAs_np=bias_AAs_np,
                        chain_M_pos=chain_M_pos,
                        omit_AA_mask=omit_AA_mask,
                        pssm_coef=pssm_coef,
                        pssm_bias=pssm_bias,
                        pssm_multi=pssm_multi,
                        pssm_log_odds_flag=bool(pssm_log_odds_flag),
                        pssm_log_odds_mask=pssm_log_odds_mask,
                        pssm_bias_flag=bool(pssm_bias_flag),
                        bias_by_res=bias_by_res_all,
                    )
                    S_sample = sample_dict["S"]
                else:
                    sample_dict = model.tied_sample(
                        X,
                        randn_2,
                        S,
                        chain_M,
                        chain_encoding_all,
                        residue_idx,
                        mask=mask,
                        temperature=temp,
                        omit_AAs_np=omit_AAs_np,
                        bias_AAs_np=bias_AAs_np,
                        chain_M_pos=chain_M_pos,
                        omit_AA_mask=omit_AA_mask,
                        pssm_coef=pssm_coef,
                        pssm_bias=pssm_bias,
                        pssm_multi=pssm_multi,
                        pssm_log_odds_flag=bool(pssm_log_odds_flag),
                        pssm_log_odds_mask=pssm_log_odds_mask,
                        pssm_bias_flag=bool(pssm_bias_flag),
                        tied_pos=tied_pos_list_of_lists_list[0],
                        tied_beta=tied_beta,
                        bias_by_res=bias_by_res_all,
                    )
                    # Compute scores
                    S_sample = sample_dict["S"]
                log_probs = model(
                    X,
                    S_sample,
                    mask,
                    chain_M * chain_M_pos,
                    residue_idx,
                    chain_encoding_all,
                    randn_2,
                    use_input_decoding_order=True,
                    decoding_order=sample_dict["decoding_order"],
                )
                mask_for_loss = mask * chain_M * chain_M_pos
                scores = _scores(S_sample, log_probs, mask_for_loss)
                scores = scores.cpu().data.numpy()

                global_scores = _scores(
                    S_sample, log_probs, mask
                )  # score the whole structure-sequence
                global_scores = global_scores.cpu().data.numpy()

                all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                all_log_probs_list.append(log_probs.cpu().data.numpy())
                S_sample_list.append(S_sample.cpu().data.numpy())
                for b_ix in range(batch_copies):
                    masked_chain_length_list = masked_chain_length_list_list[b_ix]
                    masked_list = masked_list_list[b_ix]
                    seq_recovery_rate = torch.sum(
                        torch.sum(
                            F.one_hot(S[b_ix], 21) * F.one_hot(S_sample[b_ix], 21),
                            dim=-1,
                        )
                        * mask_for_loss[b_ix]
                    ) / torch.sum(mask_for_loss[b_ix])
                    seq = S_to_seq(S_sample[b_ix], chain_M[b_ix])
                    score = scores[b_ix]
                    score_list.append(score)
                    global_score = global_scores[b_ix]
                    global_score_list.append(global_score)
                    native_seq = S_to_seq(S[b_ix], chain_M[b_ix])
                    if b_ix == 0 and j == 0 and temp == temperatures[0]:
                        start = 0
                        end = 0
                        list_of_AAs = []
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(native_seq[start:end])
                            start = end
                        native_seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                        l0 = 0
                        for mc_length in list(
                            np.array(masked_chain_length_list)[np.argsort(masked_list)]
                        )[:-1]:
                            l0 += mc_length
                            native_seq = native_seq[:l0] + "/" + native_seq[l0:]
                            l0 += 1
                        sorted_masked_chain_letters = np.argsort(masked_list_list[0])
                        print_masked_chains = [
                            masked_list_list[0][i] for i in sorted_masked_chain_letters
                        ]
                        sorted_visible_chain_letters = np.argsort(visible_list_list[0])
                        print_visible_chains = [
                            visible_list_list[0][i] for i in sorted_visible_chain_letters
                        ]
                        native_score_print = np.format_float_positional(
                            np.float32(native_score.mean()),
                            unique=False,
                            precision=4,
                        )
                        global_native_score_print = np.format_float_positional(
                            np.float32(global_native_score.mean()),
                            unique=False,
                            precision=4,
                        )

                        commit_str = __version__
                        if ca_only:
                            print_model_name = "CA_model_name"
                        else:
                            print_model_name = "model_name"
                        f.write(
                            ">{}, score={}, global_score={}, fixed_chains={}, designed_chains={}, {}={}, git_hash={}, seed={}\n{}\n".format(
                                name_,
                                native_score_print,
                                global_native_score_print,
                                print_visible_chains,
                                print_masked_chains,
                                print_model_name,
                                model_name,
                                commit_str,
                                seed,
                                native_seq,
                            )
                        )  # write the native sequence
                    start = 0
                    end = 0
                    list_of_AAs = []
                    for mask_l in masked_chain_length_list:
                        end += mask_l
                        list_of_AAs.append(seq[start:end])
                        start = end

                    seq = "".join(list(np.array(list_of_AAs)[np.argsort(masked_list)]))
                    l0 = 0
                    for mc_length in list(
                        np.array(masked_chain_length_list)[np.argsort(masked_list)]
                    )[:-1]:
                        l0 += mc_length
                        seq = seq[:l0] + "/" + seq[l0:]
                        l0 += 1
                    score_print = np.format_float_positional(
                        np.float32(score), unique=False, precision=4
                    )
                    global_score_print = np.format_float_positional(
                        np.float32(global_score), unique=False, precision=4
                    )
                    seq_rec_print = np.format_float_positional(
                        np.float32(seq_recovery_rate.detach().cpu().numpy()),
                        unique=False,
                        precision=4,
                    )
                    sample_number = j * batch_copies + b_ix + 1
                    f.write(
                        ">T={}, sample={}, score={}, global_score={}, seq_recovery={}\n{}\n".format(
                            temp,
                            sample_number,
                            score_print,
                            global_score_print,
                            seq_rec_print,
                            seq,
                        )
                    )  # write generated sequence
    if save_score:
        np.savez(
            score_file,
            score=np.array(score_list, np.float32),
            global_score=np.array(global_score_list, np.float32),
        )
    if save_probs:
        all_probs_concat = np.concatenate(all_probs_list)
        all_log_probs_concat = np.concatenate(all_log_probs_list)
        S_sample_concat = np.concatenate(S_sample_list)
        np.savez(
            probs_file,
            probs=np.array(all_probs_concat, np.float32),
            log_probs=np.array(all_log_probs_concat, np.float32),
            S=np.array(S_sample_concat, np.int32),
            mask=mask_for_loss.cpu().data.numpy(),
            chain_order=chain_list_list,
        )
    t1 = time.perf_counter()
    dt = round(float(t1 - t0), 4)
    num_seqs = len(temperatures) * num_batches * batch_copies
    total_length = X.shape[1]
    LOGGER.info(f"{num_seqs} sequences of length {total_length} generated in {dt} seconds")
