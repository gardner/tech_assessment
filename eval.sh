#!/usr/bin/env bash

set -a && source .env && set +a

# source .venv/bin/activate

export PROMPTFOO_PYTHON="$(pwd)/.venv/bin/python"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

if [ "$1" == "view" ]; then
    npx -y promptfoo@latest view --yes
else
    uv run npx -y promptfoo@latest eval -c promptfooconfig.yaml -j 4 --output evaluations/results.json
fi

