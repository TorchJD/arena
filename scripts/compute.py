import click
import pandas as pd
from tqdm import tqdm

from arena.configurations import OBJECTIVE_LISTS
from arena.interface import INTERFACES
from arena.paths import PATH_RESULTS


@click.command()
@click.argument("ref")
@click.argument("representation")
@click.argument("interface_key")
@click.argument("objectives_key")
def main(ref: str, representation: str, interface_key: str, objectives_key: str):
    interface = INTERFACES[interface_key]
    fn = interface(representation)
    column_name = f"{ref} -> {representation}"

    objectives = OBJECTIVE_LISTS[objectives_key]

    results = {}
    for objective in tqdm(objectives):
        results[f"{objective}"] = objective(fn)

    df = pd.DataFrame(list(results.items()), columns=["Objective", column_name])

    save_dir = PATH_RESULTS / objectives_key
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / f"{column_name}.csv", index=False)


if __name__ == '__main__':
    main()
