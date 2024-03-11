import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict


LOGGER = logging.getLogger(__name__)

PathLike = str | Path
SeqStr = list[str]  # actually ndarray[str]
JsonDict = dict[str, Any]  # recursive type


class PdbDict(TypedDict):
    name: str
    num_of_chains: int
    seq: str
    coords_chain: dict[str, Any]
    seq_chain: dict[str, str]


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


def try_load_jsonl(filename: PathLike | None, fail_msg: str, success_msg: Optional[str] = None ,mode: Literal["last", "update"] = "last") -> JsonDict | None:
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


# TODO: implement
def parse_pdb(
    filename: PathLike, chains: Optional[list[str]] = None, ca_only: bool = False
) -> PdbDict:
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
    return {"name": "", "num_of_chains": 1, "seq": "", "coords_chain": {}, "seq_chain": {}}


# TODO: implement
def parse_pdb_biounits(*args, **kwargs) -> tuple[None, None] | tuple[Any, list[str]]: ...


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
