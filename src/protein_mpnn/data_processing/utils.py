"""utils.py - ProteinMPNN Data Processing Utilities.

DESCRIPTION
    This module contains utility functions for processing data related to ProteinMPNN.

FUNCTIONS
    parse_pdb(file, chains, ca_only)
        Load protein sequence and structure data from a PDB file
"""

import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, TypeVar, Union

LOGGER = logging.getLogger(__name__)
PathLike = str | Path
JsonDict = dict[str, Union[str, int, list, "JsonDict"]]
# PDB id -> (designable chains, fixed chains)
ChainIdDictType = dict[str, tuple[list[str], list[str]]]
DEFAULT_COORD = (math.nan, math.nan, math.nan)


class EmptyPdbError(Exception):
    """If a PDB file has no data."""

    def __init__(self, filename: PathLike):
        """Initialize the exception."""
        super().__init__(f"Empty PDB file '{filename!s}'")


AA1 = "ARNDCQEGHILKMFPSTWYV-"
AA3 = [
        "ALA", "ARG", "ASN", "ASP", "CYS",
        "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO",
        "SER", "THR", "TRP", "TYR", "VAL", "GAP",
]  # fmt: skip
AA3_TO_NUM = {a: n for n, a in enumerate(AA3)}
AA3_TO_AA1 = dict(zip(AA3, AA1, strict=False))

PDB_LINE_RE = re.compile(
    r"""(?P<name>ATOM  |HETATM)
        (?P<serial>[ \d]{5}) (?P<atom>[ a-z]{4}).(?P<resname>[ a-z]{3})
        (?P<chain_id>[a-z0-9 ])(?P<resi>[ \d]{4})(?P<resa>.).{4}
        (?P<x>.{8})(?P<y>.{8})(?P<z>.{8})
        (?P<occ>.{6})(?P<tmp>.{6})(.{9})(?P<el>.{2})""",
    flags=re.IGNORECASE | re.VERBOSE,
)


class PdbDict(TypedDict):
    """Protein sequence and coordinates data."""

    name: str
    num_of_chains: int
    seq: str

    seq_chain_C: NotRequired[dict[str, str]]
    coords_chain_C: NotRequired[dict[str, list[tuple[float, float, float]]]]


T = TypeVar("T")


def _first_val(d: dict[str, T], default: T) -> T:
    if not d:
        return default
    return sorted(d.items())[0][1]


def parse_fasta(filename: PathLike, limit: int = -1, omit: str = "") -> tuple[list[str], list[str]]:
    """Parse a fasta sequence file into (headers, sequences).

    Args:
        filename: Path to fasta file.
        limit (optional): Number of sequences to read from file. Defaults to all.
        omit (optional): Characters to ignore from the sequence. Defaults to "".

    Returns:
        A tuple (headers, sequences), where
        headers are the sequence ids and
        sequences are the sequence contents
    """
    header: list[str] = []
    sequences: list[list[str]] = []

    with open(filename) as file:
        for line in file.readlines():
            if line.startswith(">"):
                if len(header) == limit:
                    break
                header.append(line.removeprefix(">"))
                sequences.append([])
            else:
                line = "".join(ch for ch in line if ch not in omit)
                sequences[-1].append(line)

    return header, ["".join(seq) for seq in sequences]


def try_load_jsonl(
    filename: PathLike | None,
    fail_msg: str,
    success_msg: str | None = None,
    mode: Literal["last", "update"] = "last",
) -> JsonDict | None:
    """Return a json object from a .jsonl file.

    Args:
        filename: Path to .jsonl file or None if missing.
        fail_msg: Message to log if the file cannot be loaded.
        success_msg (optional): Message to log if the file was loaded.
        mode (optional): How to load the multiple json objects.

    Returns:
        JsonDict if the data was loaded, else None
    """
    data = None
    if filename is None:
        LOGGER.debug(fail_msg)
    elif os.path.isfile(filename):
        with open(filename) as file:
            if mode == "last":
                data = json.loads(file.readlines()[-1])
            else:
                data = {}
                for obj in file.readlines():
                    data.update(json.loads(obj))
        if success_msg:
            LOGGER.info(success_msg)
    else:
        LOGGER.debug(f"File {filename!r} not found")

    return data


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


def parse_pdb_biounits(filename: PathLike, chains: str = "", atoms: list[str] | None = None):
    """Parse a PDB file into a dictionary with sequence and coordinates per chain.

    Args:
        filename: Path to PDB file.
        chains (optional): Which chains to read from file. Defaults to all.
        atoms (optional): Which atoms to use. Defaults to backbone atoms.

    Raises:
        ValueError: If no data could be extracted.

    Returns:
        A tuple (xyz, seq) where

        xyz is a dict[str, dict[str, list[tuple]]] in the format:
            {
                [chain]: {
                    [atom]: coordinates
                }
            }
        seq in a dict[str, str] in the format:
            {
                [chain]: sequence
            }
    """
    atoms = atoms or ["N", "CA", "C"]
    xyz: defaultdict[
        str,
        defaultdict[int, defaultdict[str, dict[str, tuple[float, float, float]]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    seq: defaultdict[str, defaultdict[int, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))

    with open(filename, encoding="utf-8", errors="ignore") as file:
        for line in file:
            if not (m := PDB_LINE_RE.match(line)):
                continue

            chain = m["chain_id"]
            atom = m["name"].strip()
            resname = m["resname"].strip()
            resi = int(m["resi"].strip())
            resa = m["resa"].strip()
            x, y, z = float(m["x"]), float(m["y"]), float(m["z"])

            seq[chain][resi][resa] = AA3_TO_AA1.get(resname, "-")
            xyz[chain][resi][atom][resa] = (x, y, z)

    if not xyz:
        raise EmptyPdbError(filename)

    if chains:
        for ch in list(xyz):
            if ch not in set(chains):
                del xyz[ch]
                del seq[ch]

    seq_: dict[str, str] = {}
    xyz_: dict[str, dict[str, list[tuple[float, float, float]]]] = {}

    for ch, coord_data in xyz.items():
        seq_data = seq[ch]
        min_resi, max_resi = min(coord_data), max(coord_data)

        seq_[ch] = "".join(
            [_first_val(seq_data[resi], "-") for resi in range(min_resi, max_resi + 1)],
        )
        xyz_[ch] = {
            atom: [
                _first_val(coord_data[resi][atom], DEFAULT_COORD)
                for resi in range(min_resi, max_resi + 1)
            ]
            for atom in atoms
        }

    return xyz_, seq_


def read_pdb_jsonl(
    jsonl_file: PathLike,
    truncate: int = -1,
    max_length: int = 100,
    alphabet: str = "ACDEFGHIKLMNPQRSTVWYX-",
):
    """Parse a JSONL file representing multiple PDB structures.

    Args:
        jsonl_file: Path to JSONL file.
        truncate (optional): Number of structures to read from file. Defaults to all.
        max_length (optional): Maximum fasta sequence size. Defaults to 100.
        alphabet (optional): Valid characters to read from file. Defaults to normal aminoacids.

    Returns:
        A list of PDB data dicts.
    """
    with open(jsonl_file) as file:
        entries = [json.loads(line) for line in file.readlines()]

    return filter_pdb_dicts(entries, truncate, max_length, alphabet)


def filter_pdb_dicts(
    entries: list[PdbDict],
    truncate: int = -1,
    max_length: int = 100,
    alphabet: str = "ACDEFGHIKLMNPQRSTVWYX-",
):
    """Filters a list of PDB data dicts.

    Args:
        entries: list of PDB data structures.
        truncate (optional): Number of structures to use. Defaults to all.
        max_length (optional): Maximum fasta sequence size. Defaults to 100.
        alphabet (optional): Valid characters to read. Defaults to "ACDEFGHIKLMNPQRSTVWYX-".
    """
    alphabet_set = set(alphabet)
    discard_count: defaultdict[str, int] = defaultdict(int)

    data: list[PdbDict] = []

    start = time.perf_counter()
    for i, entry in enumerate(entries, 1):
        seq = entry["seq"]
        name = entry["name"]

        bad_chars = set(seq).difference(alphabet_set)
        if bad_chars:
            LOGGER.debug(f"{name} {bad_chars} {seq}")
            discard_count["bad_chars"] += 1
        elif len(entry["seq"]) > max_length:
            discard_count["too_long"] += 1
        else:
            data.append(entry)

        # Truncate early
        if len(data) == truncate:
            return data

        if i % 1000 == 0:
            elapsed = time.perf_counter() - start
            LOGGER.debug(f"{len(data)} entries ({i} loaded) in {elapsed:.1f} s")

    LOGGER.debug(f"Discarded {discard_count}")
    return data


def get_dataset_valid(
    filename: str | None,
    chains_to_design: list[str] | str | None,
    *,
    ca_only: bool,
    max_length: int,
) -> tuple[ChainIdDictType, list[PdbDict]]:
    """Parses a .pdb or .jsonl file and marks the chains to design.

    Args:
        filename: Path to .pdb or .jsonl file.
        chains_to_design: Path to .jsonl or list of chains to design.
        ca_only: Whether to read only CA data.
        max_length: Maximum sequence lenght.

    Returns:
        tuple (chain_ids, dataset_valid) where

        chain_ids is a dict[PDB id -> (designable_chains, fixed_chains)]
        dataset_valid is a list of PDB data dictionaries
    """
    if filename is None or not Path(filename).is_file():
        raise FileNotFoundError(filename)

    if filename.lower().endswith("pdb"):
        pdb_dict = parse_pdb(filename, ca_only=ca_only)
        dataset_valid = filter_pdb_dicts(
            [pdb_dict],
            max_length=max_length,
        )
    else:
        dataset_valid = read_pdb_jsonl(
            filename,
            max_length=max_length,
        )

    if isinstance(chains_to_design, str) and Path(chains_to_design).is_file():
        chain_id_dict: dict | None = (
            try_load_jsonl(chains_to_design, "chain_id_jsonl is NOT loaded") or {}
        )

    for pdb_dict in dataset_valid:
        name = pdb_dict["name"]
        if name in chain_id_dict:
            continue

        all_chains = [
            key.removeprefix("seq_chain_") for key in pdb_dict if key.startswith("seq_chain")
        ]  # ['A','B', 'C',...]

        if chains_to_design:
            designable_chains = chains_to_design
            fixed_chains = [c for c in all_chains if c not in chains_to_design]
        else:
            designable_chains = all_chains
            fixed_chains = []

        chain_id_dict[name] = (designable_chains, fixed_chains)

    return chain_id_dict, dataset_valid
