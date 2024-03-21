"""utils.py - ProteinMPNN Data Processing Utilities.

DESCRIPTION
    This module contains utility functions for processing data related to ProteinMPNN.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar, Union

# TODO: move types to a types.py file
# TODO: use constants in the format AMINOACIDS_3TO1
# TODO: add non cannonical AA3

LOGGER = logging.getLogger(__name__)
PathLike = str | Path
JsonDict = dict[str, Union[str, int, list[Any], "JsonDict"]]
# PDB id -> (designable chains, fixed chains)
ChainIdDictType = dict[str, tuple[list[str], list[str]]]
Coord = tuple[float, float, float]

CoordinatesData = defaultdict[
    Annotated[str, "chain"],
    defaultdict[
        Annotated[int, "residue number"],
        defaultdict[Annotated[str, "atom name"], dict[Annotated[str, "residue altloc"], Coord]],
    ],
]
DEFAULT_COORD: Coord = (math.nan, math.nan, math.nan)

# NOTE: TypedDict doesn't support extra keys (see https://peps.python.org/pep-0728/)
PdbDict = dict[str, Any]


class EmptyPdbError(Exception):
    """If a PDB file has no data."""

    def __init__(self, filename: PathLike):
        """Initialize the exception."""
        super().__init__(f"Empty PDB file '{filename!s}'")


class NoValidStructureError(Exception):
    """If no structure passes the provided filters."""

    def __init__(self, filename: PathLike, filter_msg: str):
        super().__init__(f"No structure in {Path(filename).name} passes the filters {filter_msg!r}")


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
    r"""(?P<type>ATOM\s\s|HETATM)
        (?P<serial>[\s\d]{5})\s(?P<atom>[\sa-z]{4}).(?P<resname>[\sa-z]{3})\s
        (?P<chain_id>[a-z0-9\s])(?P<resi>[\s\d]{4})(?P<resa>.).{4}
        (?P<x>.{8})(?P<y>.{8})(?P<z>.{8})
        (?P<occ>.{6})(?P<tmp>.{6})(.{9})(?P<el>.{2})""",
    flags=re.IGNORECASE | re.VERBOSE,
)


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
        for line in file:
            if line.startswith(">"):
                if len(header) == limit:
                    break
                header.append(line.removeprefix(">").strip())
                sequences.append([])
            else:
                line = "".join(ch for ch in line if ch not in omit).strip()
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
                for obj in file:
                    data.update(json.loads(obj))
        if success_msg:
            LOGGER.info(success_msg)
    else:
        LOGGER.debug(f"File {filename!r} not found")

    return data


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
    xyz: CoordinatesData = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    seq: defaultdict[str, defaultdict[int, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))

    with open(filename, encoding="utf-8", errors="ignore") as file:
        for line in file:
            if not (m := PDB_LINE_RE.match(line)):
                continue

            chain = m["chain_id"]
            atom = m["atom"].strip()
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
    xyz_: dict[str, dict[str, list[Coord]]] = {}

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
        seq: str = entry["seq"]
        name: str = entry["name"]

        bad_chars = set(seq).difference(alphabet_set)
        if bad_chars:
            LOGGER.debug(f"{name} {bad_chars} {seq}")
            discard_count["bad_chars"] += 1
        elif len(seq) > max_length:
            discard_count["too_long"] += 1
        else:
            data.append(entry)

        # Truncate early
        if len(data) == truncate:
            return data

        if i % 1000 == 0:
            elapsed = time.perf_counter() - start
            LOGGER.debug(f"{len(data)} entries ({i} loaded) in {elapsed:.1f} s")

    LOGGER.debug(f"Discarded {dict(discard_count)}")
    return data
