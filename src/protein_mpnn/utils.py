import argparse
from typing import Literal


class Namespace(argparse.Namespace):
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
    omit_AAs: str
    bias_AA_jsonl: str | None
    bias_by_res_jsonl: str | None
    omit_AA_jsonl: str | None
    pssm_jsonl: str | None
    pssm_multi: float
    pssm_threshold: float
    pssm_log_odds_flag: bool
    pssm_bias_flag: bool
    tied_positions_jsonl: str | None
