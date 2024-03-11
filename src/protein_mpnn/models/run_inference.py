import copy
import logging
import random
from pathlib import Path

import numpy as np
import torch

from protein_mpnn.data_processing.utils import (
    JsonDict,
    parse_pdb,
    structure_dataset,
    structure_dataset_pdb,
    try_load_jsonl,
)
from protein_mpnn.features.build_features import tied_featurize
from protein_mpnn.models import HIDDEN_DIM, NUM_LAYERS
from protein_mpnn.models.calculate_probs import (
    calculate_conditional_probs,
    calculate_unconditional_probs,
)
from protein_mpnn.models.calculate_score import calculate_score_only
from protein_mpnn.models.generate_sequences import generate_sequences
from protein_mpnn.models.inference_model import ProteinMPNN
from protein_mpnn.models.utils import get_model
from protein_mpnn.utils import Namespace

LOGGER = logging.getLogger(__name__)

# TODO: replace with actual arguments


def get_dataset_valid(args: Namespace, chain_id_dict: JsonDict | None):
    if args.pdb_path:
        pdb_dict = parse_pdb(args.pdb_path, ca_only=args.ca_only)
        dataset_valid = structure_dataset_pdb(
            pdb_dict,
            max_length=args.max_length,
        )
        if not chain_id_dict:
            all_chain_list = [
                key.removeprefix("seq_chain_") for key in pdb_dict if key.startswith("seq_chain")
            ]  # ['A','B', 'C',...]
            if args.pdb_path_chains:
                designed_chain_list = [str(item) for item in args.pdb_path_chains.split()]
            else:
                designed_chain_list = all_chain_list
            fixed_chain_list = [
                letter for letter in all_chain_list if letter not in designed_chain_list
            ]
            chain_id_dict = {
                pdb_dict["name"]: (
                    designed_chain_list,
                    fixed_chain_list,
                )
            }
    elif args.jsonl_path:
        dataset_valid = structure_dataset(
            args.jsonl_path,
            max_length=args.max_length,
        )
    else:
        raise ValueError("Missing input file")
    return chain_id_dict, dataset_valid


# TODO: replace with actual arguments


def run_inference(args: Namespace):
    seed = args.seed or random.randint(0, 999)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    checkpoint_path = get_model(args.model_name, args.ca_only, args.use_soluble_model)

    num_batches = args.num_seq_per_target // args.batch_size
    BATCH_COPIES = args.batch_size
    omit_AAs_list = args.omit_AAs
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    alphabet_dict = dict(zip(alphabet, range(21)))
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    chain_id_dict = try_load_jsonl(args.chain_id_jsonl, "chain_id_jsonl is NOT loaded")
    fixed_positions_dict = try_load_jsonl(
        args.fixed_positions_jsonl, "fixed_positions_jsonl is NOT loaded"
    )
    omit_AA_dict = try_load_jsonl(args.omit_AA_jsonl, "omit_AA_jsonl is NOT loaded")
    bias_AA_dict = try_load_jsonl(args.bias_AA_jsonl, "bias_AA_jsonl is NOT loaded")
    tied_positions_dict = try_load_jsonl(
        args.tied_positions_jsonl, "tied_positions_jsonl is NOT loaded"
    )
    pssm_dict = try_load_jsonl(args.pssm_jsonl, "pssm_jsonl is NOT loaded", mode="update")
    bias_by_res_dict = try_load_jsonl(
        args.bias_by_res_jsonl,
        "bias by residue dictionary is not loaded, or not provided",
        "bias by residue dictionary is loaded",
    )

    LOGGER.debug(40 * "-")

    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
        for n, AA in enumerate(alphabet):
            if AA in list(bias_AA_dict.keys()):
                bias_AAs_np[n] = bias_AA_dict[AA]

    chain_id_dict, dataset_valid = get_dataset_valid(args, chain_id_dict)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    noise_level_print = checkpoint["noise_level"]
    model = ProteinMPNN(
        ca_only=args.ca_only,
        num_letters=21,
        node_features=HIDDEN_DIM,
        edge_features=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_encoder_layers=NUM_LAYERS,
        num_decoder_layers=NUM_LAYERS,
        augment_eps=args.backbone_noise,
        k_neighbors=checkpoint["num_edges"],
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    LOGGER.debug(40 * "-")
    LOGGER.debug(f"Number of edges: {checkpoint['num_edges']}")
    LOGGER.debug(f"Training noise level: {noise_level_print}A")

    # Build paths for experiment
    output_folder = Path(args.out_folder).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    # Timing
    # start_time = time.perf_counter()
    # total_residues = 0
    # protein_list = []
    # total_step = 0
    # Validation epoch
    with torch.no_grad():
        # test_sum, test_weights = 0.0, 0.0
        for protein in dataset_valid:
            score_list = []
            global_score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            (
                X,
                S,
                mask,
                lengths,
                chain_M,
                chain_encoding_all,
                chain_list_list,
                visible_list_list,
                masked_list_list,
                masked_chain_length_list_list,
                chain_M_pos,
                omit_AA_mask,
                residue_idx,
                dihedral_mask,
                tied_pos_list_of_lists_list,
                pssm_coef,
                pssm_bias,
                pssm_log_odds_all,
                bias_by_res_all,
                tied_beta,
            ) = tied_featurize(
                batch_clones,
                device,
                chain_id_dict,
                fixed_positions_dict,
                omit_AA_dict,
                tied_positions_dict,
                pssm_dict,
                bias_by_res_dict,
                ca_only=args.ca_only,
            )
            pssm_log_odds_mask = (
                pssm_log_odds_all > args.pssm_threshold
            ).float()  # 1.0 for true, 0.0 for false
            name_ = str(batch_clones[0]["name"])

            if args.score_only:
                calculate_score_only(
                    args.path_to_fasta,
                    num_batches,
                    alphabet_dict,
                    device,
                    model,
                    output_folder,
                    X,
                    S,
                    mask,
                    chain_M,
                    chain_encoding_all,
                    chain_M_pos,
                    residue_idx,
                    name_,
                )
            elif args.conditional_probs_only:
                calculate_conditional_probs(
                    args.conditional_probs_only_backbone,
                    num_batches,
                    model,
                    output_folder,
                    X,
                    S,
                    mask,
                    chain_M,
                    chain_encoding_all,
                    chain_M_pos,
                    residue_idx,
                    name_,
                )
            elif args.unconditional_probs_only:
                calculate_unconditional_probs(
                    num_batches,
                    model,
                    output_folder,
                    X,
                    S,
                    mask,
                    chain_M,
                    chain_encoding_all,
                    chain_M_pos,
                    residue_idx,
                    name_,
                )
            else:
                generate_sequences(
                    args.pssm_multi,
                    args.pssm_log_odds_flag,
                    args.pssm_bias_flag,
                    args.model_name,
                    args.ca_only,
                    args.save_score,
                    args.save_probs,
                    seed,
                    num_batches,
                    BATCH_COPIES,
                    args.sampling_temp,
                    omit_AAs_np,
                    tied_positions_dict,
                    bias_AAs_np,
                    model,
                    output_folder,
                    score_list,
                    global_score_list,
                    all_probs_list,
                    all_log_probs_list,
                    S_sample_list,
                    chain_list_list,
                    visible_list_list,
                    masked_list_list,
                    masked_chain_length_list_list,
                    tied_pos_list_of_lists_list,
                    X,
                    S,
                    mask,
                    chain_M,
                    chain_encoding_all,
                    chain_M_pos,
                    omit_AA_mask,
                    residue_idx,
                    pssm_coef,
                    pssm_bias,
                    bias_by_res_all,
                    tied_beta,
                    pssm_log_odds_mask,
                    name_,
                )
