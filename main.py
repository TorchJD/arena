"""
Example Script

Usage:
  script.py <ref> <aggregator> <objectives-key>

Arguments:
  <ref>                            TODO
  <aggregator>                     TODO
  <objectives-key>                 TODO

This script is a placeholder for demonstrating the use of docopt with two string arguments.
"""

from docopt import docopt
import pandas as pd
from pprint import pprint

from aggregator_arena.configurations import OBJECTIVE_LISTS
from torchjd.aggregation import *  # noqa
import warnings

from aggregator_arena.paths import PATH_RESULTS

warnings.filterwarnings("ignore", category=SyntaxWarning, module="docopt")


def main():
    args = docopt(__doc__)
    ref = args["<ref>"]
    aggregator_str = args["<aggregator>"]
    objectives_key = args["<objectives-key>"]

    A = eval(aggregator_str)
    column_name = f"{aggregator_str}_{ref}"

    objectives = OBJECTIVE_LISTS[objectives_key]

    results = {f"{objective}": objective(A) for objective in objectives}
    df = pd.DataFrame(list(results.items()), columns=["Objective", column_name])

    PATH_RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATH_RESULTS / f"{column_name}_{objectives_key}.csv", index=False)

    pprint(results, sort_dicts=False)


if __name__ == '__main__':
    main()
