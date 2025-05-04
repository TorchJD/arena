#!/bin/bash

show_help() {
cat <<'EOF'
Usage: ./setup_env.sh [--quiet|-q] <ref>

This script sets up a virtual environment using uv, installs the specified
version of the torchjd library from GitHub, and installs the current project in editable mode.

Arguments:
  --quiet, -q    Suppress output from uv commands
  <ref>          A Git reference of torchjd to install (can be a branch name, tag, or commit hash)

Examples:
  ./setup_env.sh main       # Install from the 'main' branch
  ./setup_env.sh -q v0.3.0  # Install from the 'v0.3.0' tag quietly
  ./setup_env.sh 194b9d     # Install from a specific commit hash
EOF
}

# Initialize quiet flag to empty and ref variable
QUIET_FLAG=""
REF=""

# Parse arguments
for arg in "$@"; do
  case $arg in
    --quiet|-q)
      QUIET_FLAG="--quiet"
      ;;
    --help|-h)
      show_help
      exit 0
      ;;
    *)
      REF="$arg"
      ;;
  esac
done

# Check if a ref was provided
if [ -z "$REF" ]; then
  echo "Error: Missing <ref> argument."
  echo "Use --help or -h for usage information."
  exit 1
fi

rm -rf .venv
uv venv $QUIET_FLAG
uv pip install $QUIET_FLAG "git+ssh://git@github.com/TorchJD/torchjd.git@$REF[full]"
uv pip install $QUIET_FLAG -e . --group check
