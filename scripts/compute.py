import os

import click
import pandas as pd
import torch
from tqdm import tqdm

from arena.interfaces import INTERFACES
from arena.objectives import OBJECTIVE_LISTS
from arena.paths import PATH_RESULTS, make_results_filename


@click.command()
@click.argument("ref")
@click.argument("representation")
@click.argument("interface_key")
@click.argument("objectives_key")
@click.argument("config_name")
def main(ref: str, representation: str, interface_key: str, objectives_key: str, config_name: str):
    # Fix randomness
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

    interface = INTERFACES[interface_key]
    fn = interface(representation)
    column_name = f"{ref} -> {representation}"

    objectives = OBJECTIVE_LISTS[objectives_key]

    results = {}
    for objective in tqdm(objectives):
        results[f"{objective}"] = objective(fn)

    df = pd.DataFrame(list(results.items()), columns=["Objective", column_name])

    save_dir = PATH_RESULTS / config_name
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / make_results_filename(ref, representation), index=False)


if __name__ == "__main__":
    main()
