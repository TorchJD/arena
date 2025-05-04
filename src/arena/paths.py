from pathlib import Path

_PATH_HERE = Path(__file__).resolve().parent
PATH_ROOT = _PATH_HERE.parent.parent
PATH_RESULTS = PATH_ROOT / "results"
PATH_CONFIGS = PATH_ROOT / "configs"


def make_results_filename(ref: str, representation: str) -> str:
    return f"{ref}->{representation}.csv"
