"""ProteinMPNN __main__ module.

DESCRIPTION
    This module serves as the entry point for running inference with the ProteinMPNN package.
    When executed directly, it calls the `run_inference` function to perform inference
    on protein sequences using a trained MPNN model.

USAGE
    $ python -m protein_mpnn
    $ protein_mpnn
"""

import logging

from protein_mpnn.cli import Namespace, parser


def main() -> None:
    """Run inference with the provided CLI arguments."""
    args = parser.parse_args(namespace=Namespace())

    from protein_mpnn.models.run_inference import run_inference
    logging.basicConfig(level=args.log_level)
    run_inference(args)


if __name__ == "__main__":
    main()
