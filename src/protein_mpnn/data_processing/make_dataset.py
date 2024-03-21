"""make_dataset.py - ProteinMPNN dataset creation module.

DESCRIPTION
    Run data processing scripts to turn raw data from into cleaned data ready to be analyzed.

FUNCTIONS
    parse_pdb(file, chains, ca_only)
        Load protein sequence and structure data from a PDB file

    get_dataset_valid(file, chains, ca_only, max_length)
        Load protein data and create a valid list of sequences to design.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Literal, Optional, cast

from protein_mpnn.data_processing.utils import (
    ChainIdDictType,
    NoValidStructureError,
    PathLike,
    PdbDict,
    filter_pdb_dicts,
    parse_pdb_biounits,
    try_load_jsonl,
)

LOGGER = logging.getLogger(__name__)


def parse_pdb(filename: PathLike, *, chains: str = "", ca_only: bool = False) -> PdbDict:
    """Parses a PDB file.

    Args:
        filename: Path to PDB file.
        chains (optional): Which chains to read from the file. Defaults to all.
        ca_only: Store only CA data? Defaults to False.

    Returns:
        A dict with PDB attributes:
            name;
            number of chains;
            fasta sequence;
            atom coordinates per chain;
            fasta sequence per chain
    """
    sidechain_atoms = ["CA"] if ca_only else ["N", "CA", "C", "O"]
    xyz, seq = parse_pdb_biounits(filename, chains, sidechain_atoms)

    pdb_dict: PdbDict = {
        "name": Path(filename).stem,
        "num_of_chains": len(seq),
        "seq": "".join(s for _, s in sorted(seq.items())),
    }

    for chain in sorted(seq):
        coords_data = {f"{atom}_chain_{chain}": xyz[chain][atom] for atom in sidechain_atoms}

        pdb_dict[f"seq_chain_{chain}"] = seq[chain]
        pdb_dict[f"coords_chain_{chain}"] = coords_data

    return pdb_dict


def get_dataset_valid(
    filename: str | None,
    chains_to_design: Optional[list[str] | str] = None,
    *,
    ca_only: bool = False,
    max_length: int,
) -> tuple[ChainIdDictType, list[PdbDict]]:
    """Parses a .pdb or .jsonl file and marks the chains to design.

    Args:
        filename: Path to .pdb or .jsonl file.
        chains_to_design (optional): Path to .jsonl or list of chains to design. Defaults to all.
        ca_only (optional): Whether to read only CA data. Defaults to False.
        max_length (optional): Maximum sequence lenght. Defaults to 100.

    Returns:
        tuple (chain_ids, dataset_valid) where

        chain_ids is a dict[PDB id -> (designable_chains, fixed_chains)]
        dataset_valid is a list of PDB data dictionaries
    """
    if filename is None or not Path(filename).exists():
        raise FileNotFoundError(filename)

    if filename.lower().endswith("pdb"):
        LOGGER.debug(f"Received .pdb file {filename}")
        pdb_dicts = [parse_pdb(filename, ca_only=ca_only)]
    elif Path(filename).is_dir():
        LOGGER.debug(f"Received directory of PDB files {filename}")
        pdb_dicts = [parse_pdb(file, ca_only=ca_only) for file in Path(filename).glob("*.pdb")]
    else:
        LOGGER.debug(f"Received .jsonl file {filename}")
        with open(filename) as file:
            pdb_dicts = [json.loads(line) for line in file]

    dataset_valid = filter_pdb_dicts(
        pdb_dicts,
        max_length=max_length,
    )

    if not dataset_valid:
        raise NoValidStructureError(filename, f"{ca_only=}, {max_length=}")

    if isinstance(chains_to_design, str) and Path(chains_to_design).is_file():
        chain_id_dict = cast(
            ChainIdDictType, try_load_jsonl(chains_to_design, "chain_id_jsonl is NOT loaded") or {}
        )
    else:
        chain_id_dict = {}

    for pdb_dict in dataset_valid:
        name: str = pdb_dict["name"]
        if name in chain_id_dict:
            continue

        all_chains = [
            key.removeprefix("seq_chain_") for key in pdb_dict if key.startswith("seq_chain")
        ]  # ['A','B', 'C',...]

        if chains_to_design:
            designable_chains = [*chains_to_design]
            fixed_chains = [c for c in all_chains if c not in chains_to_design]
        else:
            designable_chains = all_chains
            fixed_chains = []

        chain_id_dict[name] = (designable_chains, fixed_chains)

    return chain_id_dict, dataset_valid


def main(input_files: list[str], output_pdb: str, output_chains: str, chains: list[str]):
    """Creates a valid .jsonl file representing the dataset provided."""
    LOGGER.info("Reading PDB files")
    proteins: list[PdbDict] = []
    for file in input_files:
        if not Path(file).is_file():
            LOGGER.error(f"Ignoring misssing file: {file!r}")
            continue
        proteins.append(parse_pdb(file))

    chain_settings: ChainIdDictType = {}

    LOGGER.info("Defining chains to design for each file")
    for protein in proteins:
        name: str = protein["name"]
        if name in chain_settings:
            continue

        all_chains: list[str] = [
            key.removeprefix("seq_chain_") for key in protein if key.startswith("seq_chain")
        ]

        if chains:
            designable_chains = [*chains]
            fixed_chains = [c for c in all_chains if c not in chains]
        else:
            designable_chains = all_chains
            fixed_chains = []

        chain_settings[name] = (designable_chains, fixed_chains)

    LOGGER.info(f"Saving parsed PDBs to {output_pdb}")
    with open(output_pdb, "w") as handle:
        for protein in proteins:
            handle.write(json.dumps(protein))
            handle.write("\n")

    LOGGER.info(f"Saving chain ids to {output_chains}")
    with open(output_chains, "w") as handle:
        handle.write(json.dumps(chain_settings))
        handle.write("\n")


def parse_args():
    """Defines and parses CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Creates a valid .jsonl file representing the dataset provided."
    )

    class _Namespace(argparse.Namespace):
        input: list[str]
        output_pdb: str
        output_chains: str
        chains: list[str]
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    parser.add_argument("input", nargs="+", required=True, help="Path to the .PDB files")
    parser.add_argument(
        "--output_pdb",
        default="proteins.jsonl",
        help="Path to the generated .jsonl file (default: %(default)s)",
    )
    parser.add_argument(
        "--output_chains",
        default="chain_ids.jsonl",
        help="Path to the generated .jsonl file (default: %(default)s)",
    )
    parser.add_argument(
        "--chains",
        nargs="*",
        metavar="CHAIN_ID",
        default=[],
        help="List of chains to design",
        action="append",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Hide intermediate outputs",
    )

    return parser.parse_args(namespace=_Namespace())


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(format="%(message)s", level=args.log_level)
    main(args.input, args.output_pdb, args.output_chains, args.chains)
