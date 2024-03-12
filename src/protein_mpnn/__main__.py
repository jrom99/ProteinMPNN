import logging

from protein_mpnn.cli import Namespace, parser
from protein_mpnn.models.run_inference import run_inference

if __name__ == "__main__":
    args = parser.parse_args(namespace=Namespace())
    logging.basicConfig(level=args.log_level)
    run_inference(args)
