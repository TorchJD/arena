import click
import pandas as pd

from arena.paths import PATH_RESULTS

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 500)


@click.command()
@click.argument("objectives_key")
@click.argument("floatfmt", default=".2g")
def main(objectives_key: str, floatfmt: str):
    load_dir = PATH_RESULTS / objectives_key

    dfs = []
    paths = sorted(list(load_dir.iterdir()))
    for path in paths:
        df = pd.read_csv(path, index_col=0)
        dfs.append(df)

    df = pd.concat(dfs, axis=1)
    print(df.to_markdown(tablefmt="github", floatfmt=floatfmt))


if __name__ == "__main__":
    main()
