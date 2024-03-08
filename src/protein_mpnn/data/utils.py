from typing import Any, Optional, TypedDict
from pathlib import Path


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
    return [], []


def try_load_jsonl(filename: PathLike | None, fail_msg: str) -> JsonDict | None:
    """Returns a json object from a .jsonl file.

    Args:
        filename: Path to .jsonl file or None if missing.
        fail_msg: Message to log if the file cannot be loaded.

    Returns:
        JsonDict if the data was loaded, else None
    """
    return {}


def parse_pdb(filename: PathLike, chains: Optional[list[str]] = None, ca_only: bool = False) -> PdbDict:
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
    return {
        "name": "",
        "num_of_chains": 1,
        "seq": "",
        "coords_chain": {},
        "seq_chain": {}
    }


def parse_pdb_biounits(*args, **kwargs) -> tuple[None, None] | tuple[Any, list[str]]: ...
def structure_dataset(jsonl_file: PathLike, truncate) -> list: ...
def structure_dataset_pdb(*args, **kwargs) -> list: ...
