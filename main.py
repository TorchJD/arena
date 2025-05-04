import subprocess
from pathlib import Path

import click
import yaml

from arena.paths import PATH_CONFIGS, make_results_filename


@click.command()
@click.argument("cfg")
def main(cfg: str):
    config = _load_yaml(PATH_CONFIGS / f"{cfg}.yaml")
    objective_key = config["objective_key"]

    for ref in config["refs"]:
        print(f"Installing environment with torchjd at ref {_bf(_green(ref))}... ", end="")
        subprocess.check_call(["./setup_env.sh", ref, "-q"])
        print("Done.")

        interface_key = config["refs"][ref]["interface_key"]
        representations = config["refs"][ref]["representations"]
        for representation in representations:
            print(f"Computing objective values for {_bf(_purple(representation))}...")
            try:
                subprocess.check_call(["uv", "run", "-q", "python", "-m", "scripts.compute", ref, representation, interface_key, objective_key, cfg])
                print(f"Results saved at results/{objective_key}/{make_results_filename(ref, representation)}")
            except Exception as e:
                print(e)
            print()
        print()

    subprocess.check_call(["uv", "run", "-q", "python", "-m", "scripts.analyze", cfg])


def _load_yaml(file_path: Path) -> dict:
    """Load and parse a YAML file."""
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def _bf(string: str) -> str:
    """Add unicode strings for bold font and end of bold font around a string."""
    return f"\033[1m{string}\033[0m"


def _green(string: str) -> str:
    """Add unicode strings for green color and end of color around a string."""
    return f"\033[92m{string}\033[0m"


def _purple(string: str) -> str:
    """Add unicode strings for purple color and end of color around a string."""
    return f"\033[95m{string}\033[0m"


if __name__ == "__main__":
    main()
