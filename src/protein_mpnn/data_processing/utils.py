import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, NotRequired, Optional, TypedDict, TypeVar

LOGGER = logging.getLogger(__name__)

PathLike = str | Path
SeqStr = list[str]  # actually ndarray[str]
JsonDict = dict[str, Any]  # recursive type


class PdbDict(TypedDict):
    name: str
    num_of_chains: int
    seq: str

    seq_chain_C: NotRequired[dict[str, str]]
    coords_chain_C: NotRequired[dict[str, list[tuple[float, float, float]]]]


def parse_fasta(filename: PathLike, limit: int = -1, omit: str = "") -> tuple[SeqStr, SeqStr]:
    """Parses a fasta sequence file into (headers, sequences).

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

    with open(filename, "r") as file:
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
    success_msg: Optional[str] = None,
    mode: Literal["last", "update"] = "last",
) -> JsonDict | None:
    """Returns a json object from a .jsonl file.

    Args:
        filename: Path to .jsonl file or None if missing.
        fail_msg: Message to log if the file cannot be loaded.
        mode: How to load the multiple json objects.

    Returns:
        JsonDict if the data was loaded, else None
    """
    data = None
    if filename is None:
        LOGGER.debug(fail_msg)
    elif os.path.isfile(filename):
        with open(filename, "r") as file:
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


LINE_RE = re.compile(
    r"^(?P<name>ATOM  |HETATM)(?P<serial>[ \d]{5}) (?P<atom>[ a-z]{4}).(?P<resname>[ a-z]{3}) (?P<chain_id>[a-z0-9 ])(?P<resi>[ \d]{4})(?P<resa>.).{4}(?P<x>.{8})(?P<y>.{8})(?P<z>.{8})(?P<occ>.{6})(?P<tmp>.{6})(.{9})(?P<el>.{2})",
    flags=re.I,
)


def parse_pdb(filename: PathLike, chains: str = "", ca_only: bool = False) -> PdbDict:
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


AA1 = "ARNDCQEGHILKMFPSTWYV-"
AA3 = [
        "ALA", "ARG", "ASN", "ASP", "CYS",
        "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO",
        "SER", "THR", "TRP", "TYR", "VAL", "GAP"
]  # fmt: skip
AA3_TO_NUM = {a: n for n, a in enumerate(AA3)}
AA3_TO_AA1 = dict(zip(AA3, AA1))
T = TypeVar("T")


def parse_pdb_biounits(filename: PathLike, chains: str = "", atoms: Optional[list[str]] = None):
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
        str, defaultdict[int, defaultdict[str, dict[str, tuple[float, float, float]]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    seq: defaultdict[str, defaultdict[int, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))

    with open(filename, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if not (m := LINE_RE.match(line)):
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
        raise ValueError(f"Empty PDB file {filename!r}")

    if chains:
        for ch in list(xyz):
            if ch not in set(chains):
                del xyz[ch]
                del seq[ch]

    def first_val(d: dict[str, T], default: T) -> T:
        if not d:
            return default
        return sorted(d.items())[0][1]

    ANYWHERE = (math.nan, math.nan, math.nan)
    seq_: dict[str, str] = {}
    xyz_: dict[str, dict[str, list[tuple[float, float, float]]]] = {}

    for ch, coord_data in xyz.items():
        seq_data = seq[ch]
        min_resi, max_resi = min(coord_data), max(coord_data)

        seq_[ch] = "".join(
            [first_val(seq_data[resi], "-") for resi in range(min_resi, max_resi + 1)]
        )
        xyz_[ch] = {
            atom: [
                first_val(coord_data[resi][atom], ANYWHERE)
                for resi in range(min_resi, max_resi + 1)
            ]
            for atom in atoms
        }

    return xyz_, seq_


def structure_dataset(
    jsonl_file: PathLike,
    truncate: int = -1,
    max_length: int = 100,
    alphabet: str = "ACDEFGHIKLMNPQRSTVWYX-",
) -> list[JsonDict]:
    """Parse a JSONL file representing a PDB structure.

    Args:
        jsonl_file: Path to JSONL file.
        truncate (optional): Number of structures to read from file. Defaults to all.
        max_length (optional): Maximum fasta sequence size. Defaults to 100.
        alphabet (optional): Valid characters to read from file. Defaults to "ACDEFGHIKLMNPQRSTVWYX-".

    Returns:
        A list containing a parsed PDB file attributes.
    """
    alphabet_set = set(alphabet)
    discard_count = {"bad_chars": 0, "too_long": 0, "bad_seq_length": 0}

    with open(jsonl_file) as file:
        data = []

        start = time.perf_counter()
        for i, line in enumerate(file.readlines(), 1):
            entry = json.loads(line)
            seq = entry["seq"]
            name = entry["name"]

            # Convert raw coords to np arrays
            # for key, val in entry['coords'].items():
            #    entry['coords'][key] = np.asarray(val)

            # Check if in alphabet
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
        LOGGER.debug(f"discarded {discard_count}")

    return data


# TODO: implement
def structure_dataset_pdb(*args, **kwargs) -> list: ...
