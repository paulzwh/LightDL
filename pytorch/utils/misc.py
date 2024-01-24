from typing import Any
from argparse import Namespace
from pathlib import Path
import json


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
