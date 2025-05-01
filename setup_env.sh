#!/bin/bash

show_help() {
cat <<'EOF'
Usage: ./setup_env.sh <ref>

This script sets up a virtual environment using uv, installs the specified
version of the torchjd library from GitHub, and installs the current project in editable mode.

Arguments:
  <ref>    A Git reference of torchjd to install (can be a branch name, tag, or commit hash)

Examples:
  ./setup_env.sh main  # Install from the 'main' branch
  ./setup_env.sh v0.3.0  # Install from the 'v0.3.0' tag
  ./setup_env.sh 194b9d  # Install from a specific commit hash
EOF
}

# Show help if requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  show_help
  exit 0
fi

# Check if a ref was provided
if [ -z "$1" ]; then
  echo "Error: Missing <ref> argument."
  echo "Use --help or -h for usage information."
  exit 1
fi

REF="$1"

rm -rf .venv
uv venv --quiet
uv pip install --quiet "git+ssh://git@github.com/TorchJD/torchjd.git@$REF"
uv pip install --quiet -e .