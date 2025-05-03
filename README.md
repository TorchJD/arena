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
./compare_upgrad_speed.sh
```

You can create similar bash files to make other tests.

## Contributing

Before making commits, but after installing the environment, run:
```bash
uv run pre-commit install
```
