import click
import pandas as pd
from pprint import pprint

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
    column_name = f"{representation}_{ref}"

    objectives = OBJECTIVE_LISTS[objectives_key]

    results = {f"{objective}": objective(fn) for objective in objectives}
    df = pd.DataFrame(list(results.items()), columns=["Objective", column_name])

    PATH_RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATH_RESULTS / f"{column_name}_{objectives_key}.csv", index=False)

    pprint(results, sort_dicts=False)


if __name__ == '__main__':
    main()
