"""ProteinMPNN Command Line Interface (CLI).

DESCRIPTION
    This module defines a command line interface for the ProteinMPNN package using
    the argparse module. It provides a way to customize the behavior of
    ProteinMPNN via command line arguments.
"""

import argparse
from typing import Literal


class Namespace(argparse.Namespace):
    """CLI namespace after `parse_args` is called."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    ca_only: bool
    path_to_model_weights: str | None
    model_name: str
    use_soluble_model: bool
    seed: int
    save_score: bool
    save_probs: bool
    path_to_fasta: str | None
    score_only: bool
    conditional_probs_only: bool
    unconditional_probs_only: bool
    conditional_probs_only_backbone: bool
    backbone_noise: float
    num_seq_per_target: int
    batch_size: int
    max_length: int
    sampling_temp: list[float]
    out_folder: str
    pdb_path: str | None
    design_chains: list[str] | None
    jsonl_path: str | None
    chain_id_jsonl: str | None
    fixed_positions_jsonl: str | None
    omit_aas: str
    bias_aa_jsonl: str | None
    bias_by_res_jsonl: str | None
    omit_aa_jsonl: str | None
    pssm_jsonl: str | None
    pssm_multi: float
    pssm_threshold: float
    pssm_log_odds_flag: bool
    pssm_bias_flag: bool
    tied_positions_jsonl: str | None


parser = argparse.ArgumentParser()

parser.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Hide intermediate outputs",
)

parser.add_argument(
    "--ca_only",
    action="store_true",
    help="Parse CA-only structures and use CA-only models",
)
parser.add_argument("--path_to_model_weights", help="Path to model weights folder")
parser.add_argument(
    "--model_name",
    default="v_48_020",
    help="""ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030;
    v_48_010=version with 48 edges 0.10A noise (default: %(default)s)""",
)
parser.add_argument(
    "--use_soluble_model",
    action="store_true",
    help="Load ProteinMPNN weights trained on soluble proteins only.",
)

parser.add_argument(
    "--seed", type=int, default=0, help="If set to 0 then a random seed will be picked;",
)

parser.add_argument("--save_score", action="store_true", help="save score=-log_prob to npy files")
parser.add_argument(
    "--save_probs",
    action="store_true",
    help="save MPNN predicted probabilites per position",
)

parser.add_argument("--score_only", action="store_true", help="score input backbone-sequence pairs")

# TODO: check --path_to_fasta argument
parser.add_argument(
    "--path_to_fasta",
    help="""score provided input sequence in a fasta format;
    e.g. GGGGGG/PPPPS/WWW for chains A, B, C sorted alphabetically and separated by /""",
)


parser.add_argument(
    "--conditional_probs_only",
    action="store_true",
    help="output conditional probabilities p(s_i given the rest of the sequence and backbone)",
)
parser.add_argument(
    "--conditional_probs_only_backbone",
    action="store_true",
    help="if true output conditional probabilities p(s_i given backbone)",
)
parser.add_argument(
    "--unconditional_probs_only",
    action="store_true",
    help="output unconditional probabilities p(s_i given backbone) in one forward pass",
)

parser.add_argument(
    "--backbone_noise",
    type=float,
    default=0.00,
    help="Standard deviation of Gaussian noise to add to backbone atoms",
)
parser.add_argument(
    "--num_seq_per_target",
    type=int,
    default=1,
    help="Number of sequences to generate per target",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size; reduce this if running out of GPU memory",
)
parser.add_argument("--max_length", type=int, default=200000, help="Max sequence length")
parser.add_argument(
    "--sampling_temp",
    type=float,
    nargs="+",
    metavar="T",
    default=[0.1],
    help="""A list of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids.
    Suggested values 0.1, 0.15, 0.2, 0.25, 0.3.
    Higher values will lead to more diversity. (default: %(default)s)""",
)

parser.add_argument(
    "--out_folder",
    required=True,
    help="Path to a folder to output sequences, e.g. /home/out/",
)

# TODO: merge pdb_path and jsonl_path
grp = parser.add_mutually_exclusive_group(required=True)
grp.add_argument("--pdb_path", help="Path to a single PDB to be designed")
grp.add_argument("--jsonl_path", help="Path to a folder with parsed pdb into jsonl")

# TODO: merge design_chains and chain_id_jsonl
parser.add_argument(
    "--design_chains",
    nargs="+",
    metavar="C",
    default=None,
    help="Define which chains need to be designed",
)

parser.add_argument(
    "--chain_id_jsonl",
    help="""Path to a dictionary specifying which chains need to be designed
    and which ones are fixed. If not specied all chains will be designed.""",
)
parser.add_argument(
    "--fixed_positions_jsonl",
    help="Path to a dictionary with fixed positions",
)
parser.add_argument(
    "--omit_AAs",
    default="X",
    help="""Specify which amino acids should be omitted in the generated sequence,
    e.g. 'AC' would omit alanine and cystine.""",
)
parser.add_argument(
    "--bias_AA_jsonl",
    help=r"""Path to a dictionary which specifies AA composion bias if needed,
    e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.""",
)

parser.add_argument("--bias_by_res_jsonl", help="Path to dictionary with per position bias.")
parser.add_argument(
    "--omit_AA_jsonl",
    help="""Path to a dictionary which specifies which amino acids need to be omited
    from design at specific chain indices""",
)
parser.add_argument("--pssm_jsonl", help="Path to a dictionary with pssm")
parser.add_argument(
    "--pssm_multi",
    type=float,
    default=0.0,
    help="""A value between [0.0, 1.0], 0.0 means do not use pssm,
    1.0 ignore MPNN predictions (default: %(default)s)""",
)
parser.add_argument(
    "--pssm_threshold",
    type=float,
    default=0.0,
    help="A value between -inf + inf to restric per position AAs (default: %(default)s)",
)
parser.add_argument("--pssm_log_odds_flag", action="store_true")
parser.add_argument("--pssm_bias_flag", action="store_true")

parser.add_argument(
    "--tied_positions_jsonl",
    help="Path to a dictionary with tied positions",
)
