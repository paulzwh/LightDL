from typing import Any
from argparse import Namespace
from pathlib import Path
import json

import logging
import torch


class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_args(args: Namespace):
    if args.output is None:
        return
    
    with open(args.output / f"args_node{args.node_rank}.json", "w") as fp:
        json.dump(vars(args), fp, indent=4, cls=ExtendedJSONEncoder)


def count_num_params(model: torch.nn.Module, logger: logging.Logger = None):
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    if logger is not None:
        logger.info(f"# total params: {total_params}, # train params: {train_params}.")

    return total_params, train_params
