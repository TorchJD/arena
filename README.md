# arena
This repo is used to compare different functions (like aggregators or backward calls), potentially from different commits of torchjd,
on a variety of topics (computation time, precision, etc.).

## Installation

We recommend to use Python 3.12, because earlier versions of torchjd did not support Python 3.13.
```bash
uv python install 3.12.10
uv python pin 3.12.10
```

Make scripts executable:
```bash
chmod +x $(ls *.sh)
```

## Usage

Run the following command:
```bash
uv run python -m main <name>
```
With `<name>` replaced by the name of your desired configuration file, located in `configs`. For instance:
```bash
uv run python -m main upgrad_runtime.yaml
```

To make other tests, you can modify `objectives.py`, and you may need to make a new interface (the object responsible to
load your Python function and wrap it to make it have the same interface as what your objective expects) in
`interfaces.py`. You can then make a new `.yaml` configuration in `configs`.

## Contributing

Before making commits, but after installing the environment, run:
```bash
uv run pre-commit install
```
